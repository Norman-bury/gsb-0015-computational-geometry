#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task15（二维几何原语）判分脚本。

规则：
1. 交付 geom.py，暴露四个函数（点用 (x,y) 元组，坐标整数）：
   - segments_intersect(a, b, c, d) -> bool，两线段共享任意点（含端点、共线重叠）为真
   - point_in_polygon(p, polygon) -> bool，点在多边形内部或边界上为真
   - convex_hull(points) -> list，严格凸包（不含边上的共线点），逆时针
   - point_segment_distance(p, a, b) -> float，点 p 到线段 [a,b] 的欧氏距离
2. 交付命令行入口：`python3 geom.py <用例文件>`，用例文件每行一条：
   - seg ax ay bx by cx cy dx dy
   - pip px py n x1 y1 ... xn yn
   - hull n x1 y1 ... xn yn
   - dist px py ax ay bx by
   逐行输出对应结果，seg/pip 输出 0/1，hull 输出 k x1 y1 ... xk yk，dist 输出浮点。
3. 判分脚本内置精确整数参考实现，生成固定种子的用例（含退化/大坐标/近共线），
   逐函数对拍。距离用相对容差 1e-9 比较。

用法：
  python3 grade.py
  python3 grade.py --module path/to/geom.py
"""

import argparse
import importlib.util
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- 精确整数参考实现 ----------

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
            and _cross(a, b, p) == 0)


def _ref_segments_intersect(a, b, c, d):
    d1 = _cross(c, d, a)
    d2 = _cross(c, d, b)
    d3 = _cross(a, b, c)
    d4 = _cross(a, b, d)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(c, d, a):
        return True
    if d2 == 0 and _on_segment(c, d, b):
        return True
    if d3 == 0 and _on_segment(a, b, c):
        return True
    if d4 == 0 and _on_segment(a, b, d):
        return True
    return False


def _ref_point_in_polygon(p, poly):
    x, y = p
    n = len(poly)
    for i in range(n):
        if _on_segment(poly[i], poly[(i + 1) % n], p):
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            A = (y - yi) * (xj - xi)
            B = (x - xi) * (yj - yi)
            if (yj - yi) > 0:
                if A > B:
                    inside = not inside
            else:
                if A < B:
                    inside = not inside
        j = i
    return inside


def _ref_convex_hull(points):
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


def _ref_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    # 投影参数 t = dot(p-a, b-a) / |b-a|^2，用精确整数判断 t 落在区间位置
    num = (px - ax) * dx + (py - ay) * dy
    den = dx * dx + dy * dy
    if num <= 0:
        return math.hypot(px - ax, py - ay)
    if num >= den:
        return math.hypot(px - bx, py - by)
    # 垂足在线段内：距离 = |cross| / |b-a|
    cross = (px - ax) * dy - (py - ay) * dx
    return math.sqrt((cross * cross) / den)


# ---------- 用例生成 ----------

def _gen_cases():
    cases = []  # (kind, args_tuple, expected)
    rng = random.Random(12345)

    def pt(lo=-10**9, hi=10**9):
        return (rng.randint(lo, hi), rng.randint(lo, hi))

    # 1. 线段相交：随机 + 退化
    for _ in range(400):
        a, b, c, d = pt(), pt(), pt(), pt()
        cases.append(("seg", (a, b, c, d), _ref_segments_intersect(a, b, c, d)))
    # 端点相接
    cases.append(("seg", ((0, 0), (10, 0), (10, 0), (10, 10)), True))
    cases.append(("seg", ((0, 0), (10, 0), (10, 10), (10, 0)), True))
    # 共线重叠 / 共线不重叠
    cases.append(("seg", ((0, 0), (10, 0), (5, 0), (15, 0)), True))
    cases.append(("seg", ((0, 0), (10, 0), (11, 0), (20, 0)), False))
    # T 形（一端落在另一段内部）
    cases.append(("seg", ((0, 0), (10, 0), (5, -5), (5, 5)), True))
    # 平行不共线
    cases.append(("seg", ((0, 0), (10, 0), (0, 5), (10, 5)), False))
    # 近共线大坐标：方向判定必须精确（斜率相同但错开一条，不相交）
    _s1 = ((0, 0), (10**9, 10**9), (10**9, 10**9 - 1), (0, -1))
    cases.append(("seg", _s1, _ref_segments_intersect(*_s1)))
    # 真正共线大坐标（浮点会翻车）
    _s2 = ((0, 0), (10**9, 10**9), (5 * 10**8, 5 * 10**8), (2 * 10**9, 2 * 10**9))
    cases.append(("seg", _s2, _ref_segments_intersect(*_s2)))

    # 2. 点在多边形
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    for _ in range(400):
        p = pt(-20, 20)
        cases.append(("pip", (p, square), _ref_point_in_polygon(p, square)))
    cases.append(("pip", ((5, 5), square), True))        # 内部
    cases.append(("pip", ((11, 5), square), False))      # 外部
    cases.append(("pip", ((5, 0), square), True))        # 边上
    cases.append(("pip", ((0, 0), square), True))        # 顶点
    cases.append(("pip", ((5, 10), square), True))       # 上边
    # 凹多边形
    concave = [(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)]
    cases.append(("pip", ((8, 8), concave), _ref_point_in_polygon((8, 8), concave)))
    cases.append(("pip", ((5, 7), concave), _ref_point_in_polygon((5, 7), concave)))  # 凹口里
    cases.append(("pip", ((5, 4), concave), _ref_point_in_polygon((5, 4), concave)))  # 凹口下，内部
    cases.append(("pip", ((5, 5), concave), True))       # 在凹顶点上
    # 大坐标多边形 + 边上的近共线点
    big = [(0, 0), (10**9, 0), (10**9, 10**9), (0, 10**9)]
    cases.append(("pip", ((5 * 10**8, 5 * 10**8), big), True))
    cases.append(("pip", ((5 * 10**8, 0), big), True))   # 底边中点

    # 3. 凸包
    for _ in range(300):
        pts = [pt(-10**6, 10**6) for _ in range(rng.randint(1, 50))]
        cases.append(("hull", (pts,), _ref_convex_hull(pts)))
    cases.append(("hull", ([(0, 0), (1, 1), (2, 2), (3, 3)],), [(0, 0), (3, 3)]))  # 全共线
    cases.append(("hull", ([(0, 0), (0, 0), (1, 0), (1, 0), (0, 1)],), [(0, 0), (1, 0), (0, 1)]))  # 重复点
    cases.append(("hull", ([(0, 0)],), [(0, 0)]))
    cases.append(("hull", ([(0, 0), (10, 0), (10, 10), (0, 10), (5, 0), (10, 5), (5, 10), (0, 5)],),
                  [(0, 0), (10, 0), (10, 10), (0, 10)]))  # 边上共线点应被剔除

    # 4. 点到线段距离
    for _ in range(400):
        p, a, b = pt(-10**9, 10**9), pt(-10**9, 10**9), pt(-10**9, 10**9)
        cases.append(("dist", (p, a, b), _ref_distance(p, a, b)))
    cases.append(("dist", ((5, 3), (0, 0), (10, 0)), 3.0))   # 垂足在线段内
    cases.append(("dist", ((-5, 0), (0, 0), (10, 0)), 5.0))  # 垂足在端点外
    cases.append(("dist", ((15, 4), (0, 0), (10, 0)), math.hypot(5, 4)))  # 端点外
    cases.append(("dist", ((5, 0), (0, 0), (10, 0)), 0.0))   # 点在线段上
    cases.append(("dist", ((0, 0), (0, 0), (10, 0)), 0.0))   # 点即端点
    cases.append(("dist", ((5, 0), (0, 0), (0, 0)), 5.0))    # 退化线段
    _d1 = ((10**9, 10**9 + 1), (0, 0), (2 * 10**9, 2 * 10**9))
    cases.append(("dist", _d1, _ref_distance(*_d1)))  # 大坐标

    # ---------- 近共线大坐标：精确整数 cross=±1，浮点会算成 0 误判方向 ----------
    big = 10**9
    # 凸包：薄四边形，浮点会把 (big, big-1) 这个凸顶点误判为共线丢掉
    _h1 = [(0, 0), (big + 1, 0), (big + 1, big), (big, big - 1)]
    cases.append(("hull", (_h1,), _ref_convex_hull(_h1)))
    _h2 = [(0, 0), (big, big - 1), (big + 1, big), (big + 1, 0)]
    cases.append(("hull", (_h2,), _ref_convex_hull(_h2)))
    # 线段：两段近共线，相交与否取决于 cross=±1
    _s3 = ((0, 0), (big, big), (big, big - 1), (big + 1, big))
    cases.append(("seg", _s3, _ref_segments_intersect(*_s3)))
    _s4 = ((0, 0), (big, big - 1), (big, big), (big + 1, big + 1))
    cases.append(("seg", _s4, _ref_segments_intersect(*_s4)))
    # 点：极靠近（但不落）在多边形边上，浮点会误判 on_segment
    _poly = [(0, 0), (big, 0), (big, big), (0, big)]
    cases.append(("pip", ((big // 2, 1), _poly), _ref_point_in_polygon((big // 2, 1), _poly)))
    cases.append(("pip", ((1, big // 2), _poly), _ref_point_in_polygon((1, big // 2), _poly)))
    # 一条斜边上的近共线点
    _tri = [(0, 0), (big, big), (big, 0)]
    cases.append(("pip", ((big - 1, big - 2), _tri), _ref_point_in_polygon((big - 1, big - 2), _tri)))
    cases.append(("pip", ((big - 2, big - 1), _tri), _ref_point_in_polygon((big - 2, big - 1), _tri)))
    return cases


def _norm_hull(h):
    return sorted(h)


# ---------- 模块加载 ----------

def load_module(path):
    spec = importlib.util.spec_from_file_location("geom_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_fn(mod, kind, args):
    if kind == "seg":
        return bool(mod.segments_intersect(*args))
    if kind == "pip":
        return bool(mod.point_in_polygon(*args))
    if kind == "hull":
        return list(mod.convex_hull(*args))
    if kind == "dist":
        return float(mod.point_segment_distance(*args))
    raise ValueError(kind)


def _check(kind, got, want):
    if kind == "hull":
        return _norm_hull(got) == _norm_hull(want)
    if kind == "dist":
        return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)
    return got == want


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=os.path.join(HERE, "geom.py"))
    args = ap.parse_args()

    print("=" * 60)
    print("判分开始")

    if not os.path.exists(args.module):
        print("[FAIL] 找不到 geom.py（%s）" % args.module)
        sys.exit(1)

    mod = load_module(args.module)
    for fn in ["segments_intersect", "point_in_polygon", "convex_hull", "point_segment_distance"]:
        if not hasattr(mod, fn):
            print("[FAIL] geom.py 缺少函数 %s" % fn)
            sys.exit(1)

    cases = _gen_cases()
    fails = 0
    by_kind = {}
    for kind, argstuple, want in cases:
        try:
            got = _call_fn(mod, kind, argstuple)
        except Exception as e:
            fails += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if fails <= 5:
                print("[FAIL] %s %r 抛异常：%s" % (kind, argstuple, e))
            continue
        if not _check(kind, got, want):
            fails += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if fails <= 10:
                print("[FAIL] %s %r -> 得到 %r，期望 %r" % (kind, argstuple, got, want))
    total = len(cases)
    print("共 %d 个用例，错 %d 个" % (total, fails))
    for k in sorted(by_kind):
        print("  %s 错 %d 个" % (k, by_kind[k]))

    # 命令行入口冒烟测试
    cli_ok = True
    if not os.path.exists(args.module):
        cli_ok = False
    else:
        import tempfile
        sample = [
            "seg 0 0 10 0 5 -5 5 5\n",
            "pip 5 5 4 0 0 10 0 10 10 0 10\n",
            "hull 5 0 0 10 0 10 10 0 10 5 0\n",
            "dist 5 3 0 0 10 0\n",
        ]
        cf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        cf.writelines(sample)
        cf.close()
        r = subprocess.run([sys.executable, args.module, cf.name],
                           cwd=HERE, capture_output=True, text=True)
        os.unlink(cf.name)
        if r.returncode != 0:
            print("[FAIL] 命令行入口退出码非 0：%s" % r.stderr[-300:])
            cli_ok = False
        else:
            outs = r.stdout.splitlines()
            expect = ["1", "1", "4 0 0 10 0 10 10 0 10", None]
            if len(outs) < 3:
                cli_ok = False
                print("[FAIL] 命令行入口输出行数不足")
            else:
                if outs[0].strip() != "1":
                    cli_ok = False
                    print("[FAIL] 命令行 seg 判定错误：%r" % outs[0])
                if outs[1].strip() != "1":
                    cli_ok = False
                    print("[FAIL] 命令行 pip 判定错误：%r" % outs[1])
                hull_pts = _norm_hull([tuple(map(int, outs[2].split()[1:][i:i+2])) for i in range(0, len(outs[2].split()[1:]), 2)])
                if hull_pts != _norm_hull([(0, 0), (10, 0), (10, 10), (0, 10)]):
                    cli_ok = False
                    print("[FAIL] 命令行 hull 输出错误：%r" % outs[2])
                if len(outs) >= 4:
                    try:
                        d = float(outs[3])
                        if not math.isclose(d, 3.0, rel_tol=1e-9, abs_tol=1e-9):
                            cli_ok = False
                            print("[FAIL] 命令行 dist 输出错误：%r" % outs[3])
                    except Exception as e:
                        cli_ok = False
                        print("[FAIL] 命令行 dist 解析失败：%r" % outs[3])

    if cli_ok:
        print("[OK] 命令行入口冒烟测试通过")
    else:
        print("[FAIL] 命令行入口冒烟测试未通过")

    print("=" * 60)
    ok = (fails == 0 and cli_ok)
    print("总结论：", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
