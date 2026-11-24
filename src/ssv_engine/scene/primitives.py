"""Procedural meshes with position, normal and texture coordinates."""
from __future__ import annotations
import numpy as np


def cube_vertices() -> np.ndarray:
    # Each face has independent UVs so the material maps cleanly to all sides.
    faces = [
        ((0, 0, 1), [(-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)]),
        ((0, 0,-1), [(.5,-.5,-.5),(-.5,-.5,-.5),(-.5,.5,-.5),(.5,-.5,-.5),(-.5,.5,-.5),(.5,.5,-.5)]),
        ((1, 0, 0), [(.5,-.5,.5),(.5,-.5,-.5),(.5,.5,-.5),(.5,-.5,.5),(.5,.5,-.5),(.5,.5,.5)]),
        ((-1,0, 0), [(-.5,-.5,-.5),(-.5,-.5,.5),(-.5,.5,.5),(-.5,-.5,-.5),(-.5,.5,.5),(-.5,.5,-.5)]),
        ((0, 1, 0), [(-.5,.5,.5),(.5,.5,.5),(.5,.5,-.5),(-.5,.5,.5),(.5,.5,-.5),(-.5,.5,-.5)]),
        ((0,-1, 0), [(-.5,-.5,-.5),(.5,-.5,-.5),(.5,-.5,.5),(-.5,-.5,-.5),(.5,-.5,.5),(-.5,-.5,.5)]),
    ]
    data = []
    for normal, vertices in faces:
        for position in vertices:
            x, y, z = position
            # Face-local mapping; all faces occupy the full 0..1 texture square.
            if normal[0]: uv = (z + .5, y + .5)
            elif normal[1]: uv = (x + .5, z + .5)
            else: uv = (x + .5, y + .5)
            data.extend((*position, *normal, *uv))
    return np.asarray(data, dtype=np.float32)


def plane_vertices(size: float = 32.0, repeats: float = 16.0) -> np.ndarray:
    h = size / 2.0
    # UV repetition gives the floor a readable large-scale tile pattern.
    vertices = [
        ((-h,0,-h),(0,1,0),(0,0)), ((h,0,-h),(0,1,0),(repeats,0)), ((h,0,h),(0,1,0),(repeats,repeats)),
        ((-h,0,-h),(0,1,0),(0,0)), ((h,0,h),(0,1,0),(repeats,repeats)), ((-h,0,h),(0,1,0),(0,repeats)),
    ]
    return np.asarray([component for p,n,uv in vertices for component in (*p,*n,*uv)], dtype=np.float32)


def sphere_vertices(rings: int = 12, segments: int = 20) -> np.ndarray:
    """UV sphere with position, smooth normal and UV per vertex."""
    rows = []
    for y in range(rings):
        v0, v1 = y / rings, (y + 1) / rings
        lat0, lat1 = np.pi * (v0 - 0.5), np.pi * (v1 - 0.5)
        for x in range(segments):
            u0, u1 = x / segments, (x + 1) / segments
            lon0, lon1 = 2*np.pi*u0, 2*np.pi*u1
            def point(lat, lon):
                n = np.array([np.cos(lat)*np.cos(lon), np.sin(lat), np.cos(lat)*np.sin(lon)], dtype=np.float32)
                return n, (u0 if lon == lon0 else u1, 1.0-v0 if lat == lat0 else 1.0-v1)
            a, auv = point(lat0, lon0); b, buv = point(lat1, lon0)
            c, cuv = point(lat1, lon1); d, duv = point(lat0, lon1)
            for n, uv in ((a,auv),(b,buv),(c,cuv),(a,auv),(c,cuv),(d,duv)):
                rows.extend((*n, *n, *uv))
    return np.asarray(rows, dtype=np.float32)
