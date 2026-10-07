"""2D geometry primitives with exact integer orientation predicates.

All orientation / intersection / containment decisions use exact integer
arithmetic (Python big ints), so coordinates up to 1e9 (and products up to
1e18) are handled with zero floating-point error. No epsilons anywhere in
predicates. Only point_segment_distance returns a float (a length, not a
predicate).

Conventions (documented, deliberate):
- segments_intersect: touching endpoints / collinear overlap COUNT as
  intersecting (closed segments).
- point_in_polygon: points exactly ON an edge (including vertices) return
  "on" (reported as boundary; CLI prints ON, and the boolean API treats
  boundary as inside -> returns True via result != "outside").
- convex_hull: collinear points on hull edges are EXCLUDED; only extreme
  vertices are returned (Andrew monotone chain). Duplicate points removed.
"""

import sys
import math


def _cross(o, a, b):
    """Exact integer 2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, a, b):
    """True iff p lies on closed segment ab (p, a, b collinear assumed)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, p3, p4):
    """Return True iff closed segments p1p2 and p3p4 intersect.

    Endpoint touching and collinear overlap count as intersecting.
    All tests use exact integer orientation signs; degenerate (collinear)
    cases fall back to on-segment checks.
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and _on_segment(p1, p3, p4):
        return True
    if d2 == 0 and _on_segment(p2, p3, p4):
        return True
    if d3 == 0 and _on_segment(p3, p1, p2):
        return True
    if d4 == 0 and _on_segment(p4, p1, p2):
        return True
    return False


def point_in_polygon(point, polygon):
    """Classify point vs polygon.

    Returns "inside", "outside", or "on" (point exactly on an edge or
    vertex). Boundary detection is exact (integer cross == 0 + bounding
    box), so points on edges are never misclassified regardless of
    coordinate magnitude. Ray casting uses exact integer arithmetic with
    a consistent half-open edge rule to avoid vertex double-counting.
    """
    px, py = point
    n = len(polygon)
    if n < 3:
        raise ValueError("polygon must have at least 3 vertices")

    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Exact boundary test: collinear AND within bounding box.
        if (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) == 0:
            if _on_segment(point, (x1, y1), (x2, y2)):
                return "on"

        # Ray casting to +x infinity; half-open rule on y range.
        if (y1 > py) != (y2 > py):
            # x-coordinate of edge/ray intersection, compared exactly:
            # px < x1 + (x2-x1)*(py-y1)/(y2-y1), sign-safe via denominator.
            num = (x2 - x1) * (py - y1)
            den = y2 - y1
            lhs = (px - x1) * den
            if den > 0:
                hit = lhs < num
            else:
                hit = lhs > num
            if hit:
                inside = not inside
    return "inside" if inside else "outside"


def convex_hull(points):
    """Return convex hull vertices in CCW order, no repeated start point.

    Andrew's monotone chain with exact integer cross products. Collinear
    points along hull edges are excluded (cross <= 0 pops); duplicates are
    removed by sorting a set. Handles 0/1/2 unique points gracefully.
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


def point_segment_distance(point, a, b):
    """Euclidean distance from point to closed segment ab (float).

    Projection parameter is computed with exact integer arithmetic; the
    only float conversion happens on the final length, which cannot flip
    any discrete decision.
    """
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:  # degenerate segment: a == b
        return math.hypot(px - ax, py - ay)
    t_num = (px - ax) * dx + (py - ay) * dy
    if t_num <= 0:
        cx, cy = ax, ay
    elif t_num >= len_sq:
        cx, cy = bx, by
    else:
        t = t_num / len_sq
        cx = ax + t * dx
        cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _fmt_num(v):
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def run_case(tokens):
    """Dispatch one test case. Returns list of output tokens."""
    cmd = tokens[0].upper()
    vals = list(map(int, tokens[1:]))

    if cmd == "SEGMENT":
        p1, p2, p3, p4 = (vals[0:2], vals[2:4], vals[4:6], vals[6:8])
        return ["YES" if segments_intersect(p1, p2, p3, p4) else "NO"]

    if cmd == "POLYGON":
        px, py = vals[0], vals[1]
        n = vals[2]
        poly = [(vals[3 + 2 * i], vals[4 + 2 * i]) for i in range(n)]
        return [point_in_polygon((px, py), poly).upper()]

    if cmd == "HULL":
        n = vals[0]
        pts = [(vals[1 + 2 * i], vals[2 + 2 * i]) for i in range(n)]
        hull = convex_hull(pts)
        out = [str(len(hull))]
        for x, y in hull:
            out += [str(x), str(y)]
        return out

    if cmd == "DIST":
        p, a, b = (vals[0:2], vals[2:4], vals[4:6])
        return [_fmt_num(point_segment_distance(p, a, b))]

    raise ValueError(f"unknown command: {cmd}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("Input format, one case per line:")
        print("  SEGMENT x1 y1 x2 y2 x3 y3 x4 y4      -> YES|NO")
        print("  POLYGON px py n x1 y1 ... xn yn      -> INSIDE|OUTSIDE|ON")
        print("  HULL n x1 y1 ... xn yn               -> k hx1 hy1 ...")
        print("  DIST px py ax ay bx by               -> distance")
        return 1
    with open(argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            print(" ".join(run_case(line.split())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
