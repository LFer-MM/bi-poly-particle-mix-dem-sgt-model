import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from yade import O, pack, Vector3


def chord_box_3d(diameter, y, box_height, depth):
    """3D axis-aligned box inside a cylinder from chord at y through full Z depth.

    Inputs: diameter, y (clamped to ±radius), box_height, depth.
    Output: dict with x/y/z bounds, width/height/depth, min_corner and max_corner (Vector3).
    """
    r = diameter / 2.0
    y = max(-r, min(r, y))
    half_chord = math.sqrt(max(0.0, r ** 2 - y ** 2))

    x_min = -half_chord
    x_max = half_chord
    y_bottom = y
    y_top = y + box_height
    z_min = 0.0
    z_max = depth

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_bottom": y_bottom,
        "y_top": y_top,
        "z_min": z_min,
        "z_max": z_max,
        "width": x_max - x_min,
        "height": box_height,
        "depth": depth,
        "min_corner": Vector3(x_min, y_bottom, z_min),
        "max_corner": Vector3(x_max, y_top, z_max),
    }


def plot_chord_box_3d(diameter, y, box_height, depth):
    """Matplotlib 3D plot of cylinder and chord box. In: diameter, y, box_height, depth. Out: box dict from chord_box_3d."""
    box = chord_box_3d(diameter, y, box_height, depth)
    r = diameter / 2.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 120)
    z_cyl = np.linspace(0, depth, 2)
    t_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = r * np.cos(t_grid)
    y_cyl = r * np.sin(t_grid)
    ax.plot_surface(x_cyl, z_grid, y_cyl, alpha=0.08, color="teal", linewidth=0)

    for z_end in (0, depth):
        ax.plot(
            r * np.cos(theta),
            [z_end] * len(theta),
            r * np.sin(theta),
            color="teal",
            linewidth=1.2,
            alpha=0.5,
        )

    xn, xx = box["x_min"], box["x_max"]
    yn, yx = box["y_bottom"], box["y_top"]
    zn, zx = box["z_min"], box["z_max"]

    faces = [
        [(xn, zn, yn), (xx, zn, yn), (xx, zx, yn), (xn, zx, yn)],
        [(xn, zn, yx), (xx, zn, yx), (xx, zx, yx), (xn, zx, yx)],
        [(xn, zn, yn), (xx, zn, yn), (xx, zn, yx), (xn, zn, yx)],
        [(xn, zx, yn), (xx, zx, yn), (xx, zx, yx), (xn, zx, yx)],
        [(xn, zn, yn), (xn, zx, yn), (xn, zx, yx), (xn, zn, yx)],
        [(xx, zn, yn), (xx, zx, yn), (xx, zx, yx), (xx, zn, yx)],
    ]

    box_poly = Poly3DCollection(
        faces,
        alpha=0.18,
        facecolor="steelblue",
        edgecolor="steelblue",
        linewidth=0.8,
    )
    ax.add_collection3d(box_poly)

    ax.plot([0, 0], [0, depth], [0, 0], color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.quiver(xn, zn, yn - r * 0.08, xx - xn, 0, 0, color="steelblue", arrow_length_ratio=0.08, linewidth=1.2)
    ax.text(0, zn, yn - r * 0.15, f"w={box['width']:.3f}", color="steelblue", fontsize=8, ha="center")

    ax.quiver(xx + r * 0.08, zn, yn, 0, 0, yx - yn, color="coral", arrow_length_ratio=0.12, linewidth=1.2)
    ax.text(xx + r * 0.18, zn, (yn + yx) / 2, f"h={box_height:.3f}", color="coral", fontsize=8, ha="left", va="center")

    ax.quiver(xx + r * 0.08, zn, yx + r * 0.08, 0, zx - zn, 0, color="gray", arrow_length_ratio=0.06, linewidth=1.2)
    ax.text(xx + r * 0.12, depth / 2, yx + r * 0.12, f"d={depth:.3f}", color="gray", fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(
        f"Chord box 3D  —  diameter={diameter}  y={y:.3f}  h={box_height:.3f}  depth={depth:.3f}"
    )

    max_range = max(diameter, depth) / 2 * 1.2
    mid_z = depth / 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.show()

    return box


def get_surface_y(padding=0.1):
    """Max sphere top Y in current YADE scene plus padding. In: padding. Out: float (padding only if no spheres)."""
    y_max = None

    for b in O.bodies:
        if type(b.shape).__name__ != "Sphere":
            continue
        surface_y = b.state.pos[1] + b.shape.radius
        if y_max is None or surface_y > y_max:
            y_max = surface_y

    if y_max is None:
        print("  [get_surface_y] no spheres found, returning 0.0 + padding")
        return 0.0 + padding

    return y_max + padding


def ingress_random(
    diameter,
    depth,
    r_small,
    r_large,
    n_small,
    n_large,
    box_height,
    material_small,
    material_large,
    color_small,
    color_large,
    settle_steps=10000,
    padding=0.1,
    verbose=True,
):
    """Add bidisperse spheres in random mix batches via chord boxes. In: geometry, counts, materials/colors, steps. Out: None."""
    remaining_small = n_small
    remaining_large = n_large
    batch_idx = 0

    while remaining_small > 0 or remaining_large > 0:
        batch_idx += 1
        total_remaining = remaining_small + remaining_large

        if batch_idx == 1:
            valid_y = (diameter / 2.0) * -0.9
        else:
            valid_y = get_surface_y(padding=padding)

        box = chord_box_3d(
            diameter=diameter,
            y=valid_y,
            box_height=box_height,
            depth=depth,
        )

        box_vol = box["width"] * box["height"] * box["depth"]
        vol_large = (4 / 3) * math.pi * r_large ** 3
        batch_capacity = max(1, int(0.60 * box_vol / vol_large))
        batch_n = min(total_remaining, batch_capacity)

        frac_small = remaining_small / total_remaining
        n_s = min(remaining_small, round(batch_n * frac_small))
        n_l = min(remaining_large, batch_n - n_s)
        n_s = max(0, min(n_s, remaining_small))
        n_l = max(0, min(n_l, remaining_large))
        batch_n = n_s + n_l

        if verbose:
            print(
                f"[batch {batch_idx}] y_floor={valid_y:.4f}  "
                f"requesting {batch_n} ({n_s}s + {n_l}L)  "
                f"remaining before: ({remaining_small}s, {remaining_large}L)"
            )

        sp = pack.SpherePack()
        sp.makeCloud(
            minCorner=box["min_corner"],
            maxCorner=box["max_corner"],
            rMean=r_large,
            rRelFuzz=0.0,
            num=batch_n,
            periodic=False,
        )
        new_ids = sp.toSimulation(material=material_large, color=color_large, wire=False)

        actually_added = len(new_ids)
        if verbose and actually_added < batch_n:
            print(f"  WARNING: requested {batch_n}, only {actually_added} spheres were added.")

        if actually_added < batch_n and actually_added > 0:
            actual_n_s = min(remaining_small, round(actually_added * frac_small))
            actual_n_l = min(remaining_large, actually_added - actual_n_s)
            actual_n_s = max(0, min(actual_n_s, remaining_small))
            actual_n_l = max(0, min(actual_n_l, remaining_large))
        else:
            actual_n_s = n_s
            actual_n_l = n_l

        ids_to_shrink = set(random.sample(list(new_ids), actual_n_s))

        for bid in new_ids:
            b = O.bodies[bid]
            if bid in ids_to_shrink:
                b.shape.radius = r_small
                b.material = O.materials[material_small]
                b.shape.color = Vector3(color_small[0], color_small[1], color_small[2])
            else:
                b.shape.color = Vector3(color_large[0], color_large[1], color_large[2])

        remaining_small -= actual_n_s
        remaining_large -= actual_n_l

        if verbose:
            print(
                f"  actually added: {actually_added} ({actual_n_s}s + {actual_n_l}L)  "
                f"remaining after: ({remaining_small}s, {remaining_large}L)"
            )

        O.run(settle_steps, True)

    if verbose:
        print(f"Ingress complete. {batch_idx} batches.")


def ingress_segregated(
    diameter,
    depth,
    r_small,
    r_large,
    n_small,
    n_large,
    box_height,
    material_small,
    material_large,
    color_small,
    color_large,
    settle_steps=10000,
    padding=0.1,
    verbose=True,
):
    """Ingress all small spheres then all large (same params as ingress_random). In: see ingress_random. Out: None."""
    for size_label, r, n_target, material, color in (
        ("SMALL", r_small, n_small, material_small, color_small),
        ("LARGE", r_large, n_large, material_large, color_large),
    ):
        remaining = n_target
        batch_idx = 0

        while remaining > 0:
            batch_idx += 1

            if batch_idx == 1 and size_label == "SMALL":
                valid_y = (diameter / 2.0) * -0.9
            else:
                valid_y = get_surface_y(padding=padding)

            box = chord_box_3d(
                diameter=diameter,
                y=valid_y,
                box_height=box_height,
                depth=depth,
            )

            box_vol = box["width"] * box["height"] * box["depth"]
            vol_sphere = (4 / 3) * math.pi * r ** 3
            batch_capacity = max(1, int(0.60 * box_vol / vol_sphere))
            batch_n = min(remaining, batch_capacity)

            if verbose:
                print(
                    f"[{size_label} batch {batch_idx}] y_floor={valid_y:.4f}  "
                    f"requesting {batch_n}  remaining before: {remaining}"
                )

            sp = pack.SpherePack()
            sp.makeCloud(
                minCorner=box["min_corner"],
                maxCorner=box["max_corner"],
                rMean=r,
                rRelFuzz=0.0,
                num=batch_n,
                periodic=False,
            )
            new_ids = sp.toSimulation(material=material, wire=False)

            actually_added = len(new_ids)
            if verbose and actually_added < batch_n:
                print(f"  WARNING: requested {batch_n}, only {actually_added} spheres were added.")

            for bid in new_ids:
                O.bodies[bid].shape.color = Vector3(color[0], color[1], color[2])

            remaining -= actually_added

            if verbose:
                print(f"  actually added: {actually_added}  remaining after: {remaining}")

            O.run(settle_steps, True)

        if verbose:
            print(f"[{size_label}] done after {batch_idx} batches.")


def get_particle_inventory(r_small, r_large, tol=1e-6, verbose=True):
    """Classify spheres by radius vs r_small/r_large. In: radii, tol, verbose. Out: inventory dict with small/large/unclassified."""
    inventory = {
        "small": {
            "count": 0,
            "radii": set(),
            "diameters": set(),
            "materials": set(),
            "colors": set(),
        },
        "large": {
            "count": 0,
            "radii": set(),
            "diameters": set(),
            "materials": set(),
            "colors": set(),
        },
        "unclassified": 0,
    }

    for b in O.bodies:
        if type(b.shape).__name__ != "Sphere":
            continue

        r = b.shape.radius
        mat = b.material.label if b.material else "None"
        col = tuple(round(c, 3) for c in b.shape.color)

        if abs(r - r_small) <= tol:
            key = "small"
        elif abs(r - r_large) <= tol:
            key = "large"
        else:
            inventory["unclassified"] += 1
            continue

        inventory[key]["count"] += 1
        inventory[key]["radii"].add(round(r, 8))
        inventory[key]["diameters"].add(round(r * 2, 8))
        inventory[key]["materials"].add(mat)
        inventory[key]["colors"].add(col)

    if verbose:
        print("=" * 52)
        print("Particle inventory")
        print("=" * 52)
        for key in ("small", "large"):
            d = inventory[key]
            print(f"\n  {key.upper()} (r={'r_small' if key == 'small' else 'r_large'})")
            print(f"    count     : {d['count']}")
            print(f"    radii     : {d['radii']}")
            print(f"    diameters : {d['diameters']}")
            print(f"    materials : {d['materials']}")
            print(f"    colors    : {d['colors']}")

            if len(d["radii"]) > 1:
                print("    WARNING: multiple radii found — expected exactly 1")
            if len(d["materials"]) > 1:
                print("    WARNING: multiple materials found — expected exactly 1")
            if len(d["colors"]) > 1:
                print("    WARNING: multiple colors found — expected exactly 1")

        if inventory["unclassified"] > 0:
            print(f"\n  UNCLASSIFIED spheres: {inventory['unclassified']}")
            print(
                f"  (radius did not match r_small={r_small} or r_large={r_large} within tol={tol})"
            )
        print("=" * 52)

    return inventory
