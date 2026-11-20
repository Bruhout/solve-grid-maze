"""
Maze solver: grid -> graph (nodes at bends/junctions/start/end) -> Dijkstra -> colored print.

Grid codes: 0 = wall, 1 = path, 2 = start, 3 = end, 4 = highlighted shortest path.
"""
import heapq

WALL, PATH, START, END, ROUTE = 0, 1, 2, 3, 4
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

CORRIDOR_OPEN_SIDES = 2  # a straight corridor or a bend has exactly this many open neighbours
MOVE_COST = 1            # cost of moving one cell
CELL_TEXT = "  "         # two spaces = one roughly square cell

# ANSI background colors per cell value
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


def walkable(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != WALL


def is_node(grid, row, col):
    """Start/end are always nodes. Otherwise a cell is a node if it is a bend
    or junction, i.e. its open neighbours are not exactly a straight-through pair.
    Dead ends (too few open neighbours) are skipped: they can never be on a shortest path."""
    if grid[row][col] in (START, END):
        return True
    open_directions = [
        direction for direction in DIRECTIONS
        if walkable(grid, row + direction[0], col + direction[1])
    ]
    if len(open_directions) < CORRIDOR_OPEN_SIDES:
        return False
    if len(open_directions) == CORRIDOR_OPEN_SIDES:
        (first_row, first_col), (second_row, second_col) = open_directions
        return not (first_row == -second_row and first_col == -second_col)  # bend if not opposite
    return True  # 3 or 4 neighbours: junction


def build_graph(grid):
    """Convert grid into a graph. Returns (nodes dict pos->Node, start Node, end Node)."""
    nodes, start, end = {}, None, None
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if walkable(grid, row, col) and is_node(grid, row, col):
                nodes[(row, col)] = Node(row, col)
                if grid[row][col] == START:
                    start = nodes[(row, col)]
                elif grid[row][col] == END:
                    end = nodes[(row, col)]

    # Edges: from each node, walk in a straight line until we hit another node or a wall.
    for (row, col), node in nodes.items():
        for row_step, col_step in DIRECTIONS:
            scan_row, scan_col, steps = row + row_step, col + col_step, MOVE_COST
            while walkable(grid, scan_row, scan_col):
                if (scan_row, scan_col) in nodes:
                    node.edges[nodes[(scan_row, scan_col)]] = steps
                    break
                scan_row, scan_col = scan_row + row_step, scan_col + col_step
                steps += MOVE_COST
    return nodes, start, end


INFINITY = float("inf")


def dijkstra(graph, start):
    """Find the shortest distance from start to every node in the graph.

    graph: dict mapping position -> Node (as returned by build_graph)
    start: the Node to measure from

    Returns two dicts, both keyed by position:
      distances: shortest known distance from start (INFINITY if unreachable)
      previous:  the position we came from on the shortest route to each node
    """
    distances = {position: INFINITY for position in graph}
    previous = {}

    distances[start.pos] = 0
    queue = [(0, start.pos)]  # (distance, position), smallest distance pops first

    while queue:
        distance, position = heapq.heappop(queue)

        # A shorter route to this node was already found, so skip this old entry
        if distance > distances[position]:
            continue

        for neighbor, weight in graph[position].edges.items():
            new_distance = distance + weight
            if new_distance < distances[neighbor.pos]:
                distances[neighbor.pos] = new_distance
                previous[neighbor.pos] = position
                heapq.heappush(queue, (new_distance, neighbor.pos))

    return distances, previous


def build_path(graph, previous, start, end):
    """Follow the 'previous' links backwards from end to start, then flip the result."""
    position = end.pos
    path = [graph[position]]
    while position != start.pos:
        position = previous[position]
        path.append(graph[position])
    path.reverse()
    return path


def highlight_path(grid, path):
    """Return a copy of the grid with every cell along the path edges set to ROUTE
    (start and end cells keep their own colors)."""
    highlighted = [row[:] for row in grid]
    for from_node, to_node in zip(path, path[1:]):
        (from_row, from_col), (to_row, to_col) = from_node.pos, to_node.pos
        row_step = (to_row > from_row) - (to_row < from_row)
        col_step = (to_col > from_col) - (to_col < from_col)
        row, col = from_row, from_col
        while (row, col) != (to_row, to_col):
            if highlighted[row][col] == PATH:
                highlighted[row][col] = ROUTE
            row, col = row + row_step, col + col_step
        if highlighted[to_row][to_col] == PATH:
            highlighted[to_row][to_col] = ROUTE
    return highlighted


def print_maze(grid):
    """Print any maze grid (values 0-4) to the terminal using colored ANSI blocks.
    Works standalone: no solving, just display."""
    print()
    for row in grid:
        print("".join(f"{COLORS.get(value, COLORS[WALL])}{CELL_TEXT}{RESET}" for value in row))
    print()


print_grid = print_maze  # kept so solve() and older calls still work


def solve(grid, verbose=True):
    nodes, start, end = build_graph(grid)
    if start is None or end is None:
        raise ValueError("Maze needs both a start (2) and an end (3).")
    distances, previous = dijkstra(nodes, start)
    if distances[end.pos] == INFINITY:
        print("No path exists.")
        print_grid(grid)
        return None
    path = build_path(nodes, previous, start, end)
    if verbose:
        print(f"Graph has {len(nodes)} nodes. Shortest path: {distances[end.pos]} moves")
        print("Node path:", " -> ".join(str(node.pos) for node in path))
    solved = highlight_path(grid, path)
    print_grid(solved)
    return solved
