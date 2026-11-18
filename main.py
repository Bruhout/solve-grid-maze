"""
Maze solver: grid -> graph (nodes at bends/junctions/start/end) -> Dijkstra -> colored print.

Grid codes: 0 = wall, 1 = path, 2 = start, 3 = end, 4 = highlighted shortest path.
"""
import heapq

WALL, PATH, START, END, ROUTE = 0, 1, 2, 3, 4
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

# ANSI background colors per cell value (two spaces = one roughly square cell)
COLORS = {
    WALL: "\033[40m",    # black
    PATH: "\033[47m",    # white
    START: "\033[44m",   # blue
    END: "\033[41m",     # red
    ROUTE: "\033[46m",   # green
}
RESET = "\033[0m"


class Node:
    """A graph node that remembers exactly which grid cell it represents."""

    def __init__(self, row, col):
        self.pos = (row, col)       # (row, col) of the pixel this node represents
        self.edges = {}             # neighbor Node -> number of moves

    def __repr__(self):
        return f"Node{self.pos}"


def walkable(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != WALL


def is_node(grid, r, c):
    """Start/end are always nodes. Otherwise a cell is a node if it is a bend
    or junction, i.e. its open neighbours are not exactly a straight-through pair.
    Dead ends (1 open neighbour) are skipped: they can never be on a shortest path."""
    if grid[r][c] in (START, END):
        return True
    open_dirs = [d for d in DIRS if walkable(grid, r + d[0], c + d[1])]
    if len(open_dirs) < 2:
        return False
    if len(open_dirs) == 2:
        (a, b) = open_dirs
        return not (a[0] == -b[0] and a[1] == -b[1])  # bend if not opposite
    return True  # 3 or 4 neighbours: junction


def build_graph(grid):
    """Convert grid into a graph. Returns (nodes dict pos->Node, start Node, end Node)."""
    nodes, start, end = {}, None, None
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if walkable(grid, r, c) and is_node(grid, r, c):
                nodes[(r, c)] = Node(r, c)
                if grid[r][c] == START:
                    start = nodes[(r, c)]
                elif grid[r][c] == END:
                    end = nodes[(r, c)]

    # Edges: from each node, walk in a straight line until we hit another node or a wall.
    for (r, c), node in nodes.items():
        for dr, dc in DIRS:
            cr, cc, steps = r + dr, c + dc, 1
            while walkable(grid, cr, cc):
                if (cr, cc) in nodes:
                    node.edges[nodes[(cr, cc)]] = steps
                    break
                cr, cc, steps = cr + dr, cc + dc, steps + 1
    return nodes, start, end


def dijkstra(start, end):
    """Standard Dijkstra. Returns list of Nodes from start to end, or None."""
    dist = {start: 0}
    prev = {}
    counter = 0  # tie-breaker so heapq never compares Node objects
    heap = [(0, counter, start)]
    done = set()
    while heap:
        d, _, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        if node is end:
            break
        for nbr, w in node.edges.items():
            nd = d + w
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                prev[nbr] = node
                counter += 1
                heapq.heappush(heap, (nd, counter, nbr))
    if end not in dist:
        return None
    path, cur = [end], end
    while cur is not start:
        cur = prev[cur]
        path.append(cur)
    return path[::-1]


def highlight_path(grid, path):
    """Return a copy of the grid with every cell along the path edges set to 4
    (start and end cells keep their own colors)."""
    out = [row[:] for row in grid]
    for a, b in zip(path, path[1:]):
        (r1, c1), (r2, c2) = a.pos, b.pos
        dr = (r2 > r1) - (r2 < r1)
        dc = (c2 > c1) - (c2 < c1)
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if out[r][c] == PATH:
                out[r][c] = ROUTE
            r, c = r + dr, c + dc
        if out[r2][c2] == PATH:
            out[r2][c2] = ROUTE
    return out


def print_maze(grid):
    """Print any maze grid (values 0-4) to the terminal using colored ANSI blocks.
    Works standalone: no solving, just display."""
    print()
    for row in grid:
        print("".join(f"{COLORS.get(v, COLORS[WALL])}  {RESET}" for v in row))
    print()


print_grid = print_maze  # kept so solve() and older calls still work


def solve(grid, verbose=True):
    nodes, start, end = build_graph(grid)
    if start is None or end is None:
        raise ValueError("Maze needs both a start (2) and an end (3).")
    path = dijkstra(start, end)
    if path is None:
        print("No path exists.")
        print_grid(grid)
        return None
    total = sum(a.edges[b] for a, b in zip(path, path[1:]))
    if verbose:
        print(f"Graph has {len(nodes)} nodes. Shortest path: {total} moves")
        print("Node path:", " -> ".join(str(n.pos) for n in path))
    solved = highlight_path(grid, path)
    print_grid(solved)
    return solved


if __name__ == "__main__":
    example = [
        [2, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 3],
    ]
    print("Small example:")
    solve(example)

    print("\nLarger example:")
    bigger = [
        [2, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 3],
    ]
    solve(bigger)

    grid = [
        [1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3]
    ]
    print_maze(grid)
    solve(grid)
