#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""二维几何原语：全部方向判定用整数叉积，精确无 epsilon。

坐标为整数，叉积 cross(o,a,b) = (a-o) x (b-o) 是整数运算，
在 1e9 量级坐标下依然精确，避免浮点把 cross=±1 的近共线点误判为 0。
距离只在最后一步开平方时才用浮点。
"""

import math
import sys


def _cross(o, a, b):
    """向量 (a-o) 与 (b-o) 的叉积，整数精确。>0 左转，<0 右转，=0 共线。"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(a, b, p):
    """p 落在闭线段 [a,b] 上：共线且坐标被两端夹住（含端点）。"""
    return (_cross(a, b, p) == 0
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(a, b, c, d):
    """两线段是否有任意公共点。

    约定（按题意）：端点相接算相交，共线且有重叠段也算相交；
    仅共线但不重叠、或完全错开不算。
    """
    d1 = _cross(c, d, a)
    d2 = _cross(c, d, b)
    d3 = _cross(a, b, c)
    d4 = _cross(a, b, d)
    # 正常穿越：两端点严格分居对方两侧
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    # 退化情形：某端点落在对方线段上（含端点相接、T 形、共线重叠）
    if d1 == 0 and _on_segment(c, d, a):
        return True
    if d2 == 0 and _on_segment(c, d, b):
        return True
    if d3 == 0 and _on_segment(a, b, c):
        return True
    if d4 == 0 and _on_segment(a, b, d):
        return True
    return False


def point_in_polygon(p, polygon):
    """p 在多边形内部或边界上即真。

    先查是否在任一条边（含顶点）上——边上的点、共线点直接判真；
    再用射线法，跨越判定全部用整数比较，y 恰好等于顶点高度时
    用半开区间 (yi > y) != (yj > y) 保证顶点不被重复计数。
    多边形顶点按顺序排列即可，顺/逆时针均可。
    """
    n = len(polygon)
    if n == 0:
        return False
    x, y = p
    for i in range(n):
        if _on_segment(polygon[i], polygon[(i + 1) % n], p):
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            # 比较 x 与边在高度 y 处的交点横坐标，交叉相乘保持整数精确
            lhs = (y - yi) * (xj - xi)
            rhs = (x - xi) * (yj - yi)
            if yj - yi > 0:
                if lhs > rhs:
                    inside = not inside
            else:
                if lhs < rhs:
                    inside = not inside
        j = i
    return inside


def convex_hull(points):
    """严格凸包，逆时针返回；边上共线点剔除，重复点去重。

    Andrew 单调链：出栈条件 cross <= 0，共线的中间点被弹出；
    全部点共线时只剩两个端点；不足两个点时原样返回去重结果。
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_segment_distance(p, a, b):
    """p 到闭线段 [a,b] 的欧氏距离（float）。

    投影参数 t = dot(p-a, b-a)/|b-a|^2，用整数比较分三段：
    t<=0 取到 a 的距离；t>=1 取到 b 的距离；中间用 |cross|/|b-a|。
    a==b 退化成点到点距离。只在开平方时引入浮点。
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    num = (px - ax) * dx + (py - ay) * dy
    if num <= 0:
        return math.hypot(px - ax, py - ay)
    den = dx * dx + dy * dy
    if num >= den:
        return math.hypot(px - bx, py - by)
    cross = (px - ax) * dy - (py - ay) * dx
    return math.sqrt((cross * cross) / den)


def _main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: python3 geom.py <case-file>\n")
        return 1
    out = []
    with open(argv[1], "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            vals = list(map(int, parts[1:]))
            if kind == "seg":
                ax, ay, bx, by, cx, cy, dx, dy = vals
                r = segments_intersect((ax, ay), (bx, by), (cx, cy), (dx, dy))
                out.append("1" if r else "0")
            elif kind == "pip":
                px, py, n = vals[0], vals[1], vals[2]
                poly = [(vals[3 + 2 * i], vals[4 + 2 * i]) for i in range(n)]
                r = point_in_polygon((px, py), poly)
                out.append("1" if r else "0")
            elif kind == "hull":
                n = vals[0]
                pts = [(vals[1 + 2 * i], vals[2 + 2 * i]) for i in range(n)]
                hull = convex_hull(pts)
                flat = [str(v) for xy in hull for v in xy]
                out.append(str(len(hull)) + (" " + " ".join(flat) if flat else ""))
            elif kind == "dist":
                px, py, ax, ay, bx, by = vals
                r = point_segment_distance((px, py), (ax, ay), (bx, by))
                out.append(repr(r))
            else:
                sys.stderr.write("unknown command: %s\n" % kind)
                return 1
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
