#!/usr/bin/env python3
"""
gpd.py — minimal GPD helpers for bt_pick_place.py

GPD orientation matrix columns (hand_set.cpp):
    col 0  approach direction  (toward object)
    col 1  binormal
    col 2  hand axis           (finger closing)

Panda TCP axes:
    x  finger closing   →  GPD col 2
    y  -binormal        → -GPD col 1
    z  approach         →  GPD col 0

So:  R_panda = [ col2 | -col1 | col0 ]
"""

import ctypes
from dataclasses import dataclass

import numpy as np
import tf_transformations
from geometry_msgs.msg import Point, Pose, Quaternion


# ── ctypes structs ────────────────────────────────────────────────────────────

class _GraspStruct(ctypes.Structure):
    _fields_ = [
        ('pos',    ctypes.POINTER(ctypes.c_double)),  # [3]  position
        ('orient', ctypes.POINTER(ctypes.c_double)),  # [4]  quaternion x,y,z,w
        ('sample', ctypes.POINTER(ctypes.c_double)),  # [3]  sample point
        ('score',  ctypes.c_double),
        ('label',  ctypes.c_bool),                    # is_full_antipodal
        ('image',  ctypes.POINTER(ctypes.c_int)),
    ]


# ── GPD interface ─────────────────────────────────────────────────────────────

class GPDInterface:
    """Thin ctypes wrapper around libgpd_python.so."""

    def __init__(self, lib_path: str):
        lib = ctypes.CDLL(lib_path)

        fn = lib.detectGraspsInFile
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_char_p,                           # config
            ctypes.c_char_p,                           # pcd
            ctypes.c_char_p,                           # normals (None ok)
            ctypes.POINTER(ctypes.c_float),            # view_points
            ctypes.c_int,                              # num_view_points
            ctypes.POINTER(ctypes.POINTER(_GraspStruct)),
        ]
        self._detect = fn

        free_fn = lib.freeMemoryGrasps
        free_fn.restype = ctypes.c_int
        free_fn.argtypes = [ctypes.POINTER(_GraspStruct)]
        self._free = free_fn

    def detect(self, config: str, pcd: str, camera_pos=(0.0, 0.0, 1.0)):
        """
        Run GPD on *pcd* and return candidates sorted by score (descending).

        Returns list of dicts: {'pos': ndarray[3], 'R': ndarray[3,3],
                                 'score': float, 'antipodal': bool}
        """
        vp = (ctypes.c_float * 3)(*camera_pos)
        grasps_ptr = ctypes.POINTER(_GraspStruct)()

        n = self._detect(
            config.encode(), pcd.encode(), None,
            vp, 1, ctypes.byref(grasps_ptr),
        )

        results = []
        for i in range(n):
            g = grasps_ptr[i]
            pos = np.array([g.pos[0], g.pos[1], g.pos[2]])
            R = tf_transformations.quaternion_matrix(
                [g.orient[0], g.orient[1], g.orient[2], g.orient[3]]
            )[:3, :3]
            results.append({
                'pos': pos, 'R': R,
                'score': g.score, 'antipodal': bool(g.label),
            })

        if n > 0:
            self._free(grasps_ptr)

        results.sort(key=lambda g: g['score'], reverse=True)
        return results


# ── PCD helpers ───────────────────────────────────────────────────────────────

def write_pcd_ascii(points: np.ndarray, path: str):
    """Write an (N,3) float32 array as an ASCII PCD file."""
    n = len(points)
    with open(path, 'w') as f:
        f.write('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n'
                'SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
                f'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                f'POINTS {n}\nDATA ascii\n')
        for p in points:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


# ── Pose conversion ───────────────────────────────────────────────────────────

def gpd_to_panda_pose(pos: np.ndarray, R_gpd: np.ndarray,
                      depth_offset: float = 0.03) -> Pose:
    """Convert a GPD grasp (pos + R) to a Panda TCP Pose."""
    col0, col1, col2 = R_gpd[:, 0], R_gpd[:, 1], R_gpd[:, 2]
    R_panda = np.column_stack([col2, -col1, col0])
    tcp_pos = pos + depth_offset * col0

    T = np.eye(4)
    T[:3, :3] = R_panda
    T[:3, 3] = tcp_pos
    qx, qy, qz, qw = tf_transformations.quaternion_from_matrix(T)

    p = Pose()
    p.position = Point(x=float(tcp_pos[0]), y=float(tcp_pos[1]), z=float(tcp_pos[2]))
    p.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    return p


# ── Module-level instance ─────────────────────────────────────────────────────

GPD_LIB    = '/root/ws/src/panda_gz_moveit2/deps/gpd/build/libgpd_python.so'
GPD_CONFIG = '/root/ws/src/panda_gz_moveit2/deps/gpd/cfg/eigen_params.cfg'
_PCD_PATH  = '/tmp/gpd_cloud.pcd'

try:
    gpd_instance = GPDInterface(GPD_LIB)
except Exception as e:
    gpd_instance = None
    print(f'[WARN] GPD unavailable: {e}')


# ── API ───────────────────────────────────────────────────────────────────────

@dataclass
class GraspCandidate:
    pos:   np.ndarray  # (3,) position in world frame
    R:     np.ndarray  # (3,3) rotation matrix — col 0: approach, col 1: binormal, col 2: hand axis
    score: float


def sample_cuboid_surface(center, dims, n_points: int = 2000,
                          orientation=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    """Sample n_points uniformly on the surface of a cuboid.

    orientation : (qx, qy, qz, qw) — object orientation in world frame.
    Points are sampled in the object's local frame, rotated, then translated.
    """
    dx, dy, dz = dims
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    Axy, Axz, Ayz = dx * dy, dx * dz, dy * dz
    A_total = 2.0 * (Axy + Axz + Ayz)

    parts = []
    for face_dims, axis, val in [
        ((hx, hy), 2,  hz), ((hx, hy), 2, -hz),
        ((hx, hz), 1,  hy), ((hx, hz), 1, -hy),
        ((hy, hz), 0,  hx), ((hy, hz), 0, -hx),
    ]:
        n = max(4, int(n_points * face_dims[0] * face_dims[1] * 4.0 / A_total))
        u = np.random.uniform(-face_dims[0], face_dims[0], n)
        v = np.random.uniform(-face_dims[1], face_dims[1], n)
        pts = np.zeros((n, 3))
        if axis == 2:   pts[:, 0] = u; pts[:, 1] = v; pts[:, 2] = val
        elif axis == 1: pts[:, 0] = u; pts[:, 1] = val; pts[:, 2] = v
        else:           pts[:, 0] = val; pts[:, 1] = u; pts[:, 2] = v
        parts.append(pts)

    pts = np.vstack(parts).astype(np.float32)

    # Rotate into world frame
    R = tf_transformations.quaternion_matrix(orientation)[:3, :3]
    pts = (R @ pts.T).T

    cx, cy, cz = center
    pts[:, 0] += cx; pts[:, 1] += cy; pts[:, 2] += cz
    return pts


def detect_grasps(cloud: np.ndarray) -> list:
    """Run GPD on a point cloud and return ranked GraspCandidates (best-first).

    cloud : (N, 3) float32 array in world frame.
    """
    write_pcd_ascii(cloud, _PCD_PATH)
    return [
        GraspCandidate(pos=c['pos'], R=c['R'], score=c['score'])
        for c in gpd_instance.detect(GPD_CONFIG, _PCD_PATH)
    ]
