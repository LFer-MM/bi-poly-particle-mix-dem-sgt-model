# IMPORTACION DE MODULOS -------------------------------------------------------------------------------------------------||

## LIBRERIA ESTANDAR
import os
import csv
from math import radians, pi

## LOCALES
from yade import ymport, Vector3, pack, Sphere, FrictMat
from yade.utils import facet, unbalancedForce, sphere, wall
from yade._polyhedra_utils import PWaveTimeStep
from yade.wrapper import (O, ForceResetter, InsertionSortCollider, Bo1_Sphere_Aabb, Bo1_Facet_Aabb,
                          InteractionLoop, Ig2_Sphere_Sphere_ScGeom, Ig2_Facet_Sphere_ScGeom,
                          Ip2_FrictMat_FrictMat_FrictPhys, Ip2_FrictMat_FrictMat_MindlinPhys,
                          Law2_ScGeom_FrictPhys_CundallStrack, Law2_ScGeom_MindlinPhys_Mindlin,
                          NewtonIntegrator, RotationEngine, PyRunner)

# ESTRUCTURAS DE APOYO -------------------------------------------------------------------------------------------------||

BALANCE_STATE = {
    "done": False,
    "threshold": 1e-3,
    "label": "balance_monitor",
    "last_unb": None,
}

MATERIALS_MAP = {}

SAG_MILL_SLICE_BODY_GROUP = None

_frameCaptureState = {}

# FUNCIONES PRINCIPALES ------------------------------------------------------------------------------------------------||

def initialize_simulation_materials(materials):
    """Register FrictMat entries from dict values into O.materials and MATERIALS_MAP. In: materials dict. Out: None."""
    for material_properties in materials.values():
        m = FrictMat(density=material_properties["density"],
                     young=material_properties["young"],
                     poisson=material_properties["poisson"],
                     frictionAngle=radians(material_properties["frictionAngle"]),
                     label=material_properties["label"])

        O.materials.append(m)
        MATERIALS_MAP[material_properties["label"]] = m
        print("Added material:", material_properties["label"])

def initialize_sag_mill_slice(sagmill_stl_path):
    """Load STL slice, add end caps, set SAG_MILL_SLICE_BODY_GROUP. In: path str. Out: None."""
    global SAG_MILL_SLICE_BODY_GROUP

    sag_mill_stl = ymport.stl(sagmill_stl_path, color=(1,1,1), wire=False, material="steel")
    sag_mill_slice_body_group = O.bodies.append(sag_mill_stl)
    print("SAG Mill Slice bodies added. Current quantity:", len(sag_mill_slice_body_group))

    sag_mill_slice_radius_m, z_min, z_max = _obtain_sag_mill_slice_measurements( sag_mill_slice_body_group)
    _add_sag_mill_slice_caps(sag_mill_slice_body_group, sag_mill_slice_radius_m, z_min, z_max)
    print("Added caps to SAG Mill Slice. Current body quantity:", len(sag_mill_slice_body_group))

    SAG_MILL_SLICE_BODY_GROUP = sag_mill_slice_body_group

def initialize_engines(contact_model, contact_model_params, rotation_engine=False):
    """Build O.engines for Cundall–Strack or Hertz–Mindlin; optionally append RotationEngine. In: model name, params dict, flag. Out: None."""
    if contact_model == "cundall_strack":
        O.engines = [
            ForceResetter(),
            InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb()], verletDist=.02),
            InteractionLoop(
                [Ig2_Sphere_Sphere_ScGeom(),Ig2_Facet_Sphere_ScGeom()],
                [Ip2_FrictMat_FrictMat_FrictPhys()],
                [Law2_ScGeom_FrictPhys_CundallStrack()],
            ),
            NewtonIntegrator(damping=0, gravity=(0,-9.81,0), label="newton_integrator"),
        ]

    if contact_model == "hertz_mindlin":
        O.engines = [
            ForceResetter(),
            InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb()], verletDist=.02),
            InteractionLoop(
                [Ig2_Sphere_Sphere_ScGeom(),
                Ig2_Facet_Sphere_ScGeom()],
                [Ip2_FrictMat_FrictMat_MindlinPhys(
                    en = contact_model_params["restitution"]
                )],
                [Law2_ScGeom_MindlinPhys_Mindlin()]
            ),
            NewtonIntegrator(damping=0, gravity=(0,-9.81,0), label="newton_integrator"),
        ]

    if rotation_engine:
        O.engines += [RotationEngine(rotateAroundZero=True, zeroPoint=(0,0,0), rotationAxis=(0,0,1), angularVelocity=0, ids=SAG_MILL_SLICE_BODY_GROUP, label="rotation_engine")]

    print("Initialized engines:", O.engines)

def load_rock_particles(rock_diam_m, rock_count):
    """Spawn rock spheres in four regions. In: diameter (m), count. Out: None (prints)."""
    rock_sphere_pack_0 = pack.SpherePack()
    rock_sphere_pack_0.makeCloud(minCorner=(-4.5, 1, 0), maxCorner=(4.5, -2, 0.375), rMean=rock_diam_m/2, rRelFuzz=0, num=int(rock_count*0.5))
    rock_sphere_pack_0.toSimulation(material="rock", color=(1,0,0), wire=False)

    rock_sphere_pack_1 = pack.SpherePack()
    rock_sphere_pack_1.makeCloud(minCorner=(-4, -2, 0), maxCorner=(4, -3, 0.375), rMean=rock_diam_m/2, rRelFuzz=0, num=int(rock_count*0.25))
    rock_sphere_pack_1.toSimulation(material="rock", color=(1,0,0), wire=False)

    rock_sphere_pack_2 = pack.SpherePack()
    rock_sphere_pack_2.makeCloud(minCorner=(-3, -3, 0), maxCorner=(3, -4, 0.375), rMean=rock_diam_m/2, rRelFuzz=0, num=int(rock_count*0.15))
    rock_sphere_pack_2.toSimulation(material="rock", color=(1,0,0), wire=False)

    rock_sphere_pack_3 = pack.SpherePack()
    rock_sphere_pack_3.makeCloud(minCorner=(-2, -4, 0), maxCorner=(2, -5, 0.375), rMean=rock_diam_m/2, rRelFuzz=0, num=int(rock_count*0.1))
    rock_sphere_pack_3.toSimulation(material="rock", color=(1,0,0), wire=False)

    message = "Added %s rock particles of %s m diameter to simulation." % (rock_count, rock_diam_m)
    print(message)

def load_ball_particles(ball_diam_m, ball_count):
    """Spawn ball_steel spheres in four regions. In: diameter (m), count. Out: None (prints)."""
    ball_sphere_pack_0 = pack.SpherePack()
    ball_sphere_pack_0.makeCloud(minCorner=(-2.75, 4, 0), maxCorner=(2.75, 5, 0.375), rMean=ball_diam_m/2, rRelFuzz=0, num=int(ball_count*0.1))
    ball_sphere_pack_0.toSimulation(material="ball_steel", color=(0,0,1), wire=False)

    ball_sphere_pack_1 = pack.SpherePack()
    ball_sphere_pack_1.makeCloud(minCorner=(-3.75, 2.5, 0), maxCorner=(3.75, 4, 0.375), rMean=ball_diam_m/2, rRelFuzz=0, num=int(ball_count*0.15))
    ball_sphere_pack_1.toSimulation(material="ball_steel", color=(0,0,1), wire=False)

    ball_sphere_pack_2 = pack.SpherePack()
    ball_sphere_pack_2.makeCloud(minCorner=(-4.25, 0, 0), maxCorner=(4.25, 2.5, 0.375), rMean=ball_diam_m/2, rRelFuzz=0, num=int(ball_count*0.25))
    ball_sphere_pack_2.toSimulation(material="ball_steel", color=(0,0,1), wire=False)

    ball_sphere_pack_3 = pack.SpherePack()
    ball_sphere_pack_3.makeCloud(minCorner=(-4.5, -3, 0), maxCorner=(4.5, 0, 0.375), rMean=ball_diam_m/2, rRelFuzz=0, num=int(ball_count*0.5))
    ball_sphere_pack_3.toSimulation(material="ball_steel", color=(0,0,1), wire=False)

    message = "Added %s ball particles of %s m diameter to simulation." % (ball_count, ball_diam_m)
    print(message)

def load_all_particles(particle_diam_m, particle_count):
    """Spawn white ball_steel particles across vertical stack regions. In: diameter, count. Out: None."""
    particle_sphere_pack_0 = pack.SpherePack()
    particle_sphere_pack_0.makeCloud(minCorner=(-2.75, 4, 0), maxCorner=(2.75, 5, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.05))
    particle_sphere_pack_0.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_1 = pack.SpherePack()
    particle_sphere_pack_1.makeCloud(minCorner=(-3.75, 3, 0), maxCorner=(3.75, 4, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.1))
    particle_sphere_pack_1.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_2 = pack.SpherePack()
    particle_sphere_pack_2.makeCloud(minCorner=(-4.25, 2, 0), maxCorner=(4.25, 3, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.15))
    particle_sphere_pack_2.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_4 = pack.SpherePack()
    particle_sphere_pack_4.makeCloud(minCorner=(-4.5, 0, 0), maxCorner=(4.5, 2, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.2))
    particle_sphere_pack_4.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_5 = pack.SpherePack()
    particle_sphere_pack_5.makeCloud(minCorner=(-4.5, 0, 0), maxCorner=(4.5, -2, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.2))
    particle_sphere_pack_5.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_6 = pack.SpherePack()
    particle_sphere_pack_6.makeCloud(minCorner=(-4.25, -2, 0), maxCorner=(4.25, -3, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.15))
    particle_sphere_pack_6.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_7 = pack.SpherePack()
    particle_sphere_pack_7.makeCloud(minCorner=(-3.75, -3, 0), maxCorner=(3.75, -4, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.1))
    particle_sphere_pack_7.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    particle_sphere_pack_8 = pack.SpherePack()
    particle_sphere_pack_8.makeCloud(minCorner=(-2.75, -4, 0), maxCorner=(2.75, -5, 0.375), rMean=particle_diam_m/2, rRelFuzz=0, num=int(particle_count*0.05))
    particle_sphere_pack_8.toSimulation(material="ball_steel", color=(1,1,1), wire=False)

    message = "Added %s particles of %s m diameter to simulation." % (particle_count, particle_diam_m)
    print(message)

def set_dt(new_dt=None, factor=0.3):
    """Set simulation timestep explicitly or from P-wave factor. In: new_dt or factor. Out: None (prints O.dt)."""
    if new_dt: O.dt = new_dt
    else: O.dt = factor * PWaveTimeStep()
    print("O.dt set to: ", O.dt)

def set_gravity_damping(new_gravity_damping):
    """Set NewtonIntegrator damping by label. In: damping float. Out: None."""
    newton_integrator = next(e for e in O.engines if getattr(e, "label", None) == "newton_integrator")
    newton_integrator.damping = new_gravity_damping
    print("Newton Integrator Gravity Damping set to: ", newton_integrator.damping)

def save_particle_positions(csv_path, include_velocity=True, include_ang_vel=True):
    """Write sphere states to CSV. In: path, velocity/angular flags. Out: csv_path str."""
    header = ["id", "x", "y", "z", "r", "m"]

    if include_velocity:
        header += ["vx", "vy", "vz"]

    if include_ang_vel:
        header += ["wx", "wy", "wz"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for b in O.bodies:
            if not b:
                continue
            if not isinstance(getattr(b, "shape", None), Sphere):
                continue

            p = b.state.pos
            r = float(b.shape.radius)
            m = _mat_label(b)

            row = [int(b.id), float(p[0]), float(p[1]), float(p[2]), r, m]

            if include_velocity:
                v = b.state.vel
                row += [float(v[0]), float(v[1]), float(v[2])]

            if include_ang_vel:
                om = b.state.angVel
                row += [float(om[0]), float(om[1]), float(om[2])]

            w.writerow(row)

    print("Saved particle positions to path:", csv_path)
    return csv_path

def load_particle_positions(csv_path, *, set_vel_zero = True, set_ang_vel_zero = True):
    """Recreate spheres from CSV using MATERIALS_MAP. In: path, zero-vel flags. Out: list of body ids."""
    created_ids = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required = {"x", "y", "z", "r", "m"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        has_v = {"vx", "vy", "vz"}.issubset(reader.fieldnames)
        has_w = {"wx", "wy", "wz"}.issubset(reader.fieldnames)

        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            r = float(row["r"])
            mat_label = (row.get("m", "") or "").strip()

            if mat_label not in MATERIALS_MAP:
                raise KeyError(
                    f"Material '{mat_label}' not found in material_map. "
                    f"Available: {list(MATERIALS_MAP.keys())}"
                )

            mat = MATERIALS_MAP[mat_label]

            if mat_label == "rock":
                sph_color = (1,0,0)
            else:
                sph_color = (0,0,1)

            # Create sphere
            bid = O.bodies.append(sphere((x, y, z), r, material=mat, color=sph_color))
            created_ids.append(bid)

            b = O.bodies[bid]

            # Velocities
            if set_vel_zero or not has_v:
                b.state.vel = Vector3(0, 0, 0)
            else:
                b.state.vel = Vector3(
                    float(row["vx"]),
                    float(row["vy"]),
                    float(row["vz"]),
                )

            if set_ang_vel_zero or not has_w:
                b.state.angVel = Vector3(0, 0, 0)
            else:
                b.state.angVel = Vector3(
                    float(row["wx"]),
                    float(row["wy"]),
                    float(row["wz"]),
                )

    return created_ids


def run_until_forces_balanced(threshold=0.001, interval=1000, motion_start_steps=20, wait_chunk=1000, max_chunks=5000):
    """Run until unbalancedForce < threshold or raise. In: thresholds/intervals. Out: True if balanced."""
    print("Starting balanced forces monitoring ...")

    BALANCE_STATE["done"] = False
    BALANCE_STATE["threshold"] = threshold

    for e in list(O.engines):
        if getattr(e, "label", None) == BALANCE_STATE["label"]:
            O.engines.remove(e)

    O.run(motion_start_steps, True)

    O.engines += [PyRunner(iterPeriod=interval, command="s1_sim_functions._balance_check()", label=BALANCE_STATE["label"])]

    for k in range(max_chunks):
        if BALANCE_STATE["done"]:
            return True
        O.run(wait_chunk, True)

    for e in list(O.engines):
        if getattr(e, "label", None) == BALANCE_STATE["label"]:
            O.engines.remove(e)

    raise RuntimeError(f"Forces didn't balance. LAST UNBALANCED FORCE = {BALANCE_STATE.get('last_unb')} | THRESHOLD: {threshold}")

def settle_balance_save(gravity_damping, csv_path):
    """Set damping, run_until_forces_balanced, save_particle_positions. In: damping, csv_path. Out: None."""
    set_gravity_damping(gravity_damping)
    run_until_forces_balanced()
    save_particle_positions(csv_path)

def rotate_mill_indefinitely(speed_rpm=9):
    """Set rotation_engine angular velocity from RPM and O.run() open-ended. In: speed_rpm. Out: None."""
    rotation_engine = next(e for e in O.engines if getattr(e, "label", None) == "rotation_engine")
    rotation_engine.angularVelocity = float(speed_rpm) * (2*pi) / 60
    O.run()

def rotate_mill_by_degrees(degrees, speed_rpm=9):
    """Rotate mill for time matching |degrees| at given RPM. In: degrees, speed_rpm. Out: None."""
    rotation_engine = _get_rotation_engine("rotation_engine")

    rpm = abs(float(speed_rpm))
    omega = rpm * (2*pi) / 60.0  # rad/s

    direction = 1.0 if degrees > 0 else -1.0
    rotation_engine.angularVelocity = direction * omega

    t_needed = (abs(float(degrees)) / 360.0) * (60.0 / rpm)  # SEGUNDOS VIRTUALES
    n_steps = int(round(t_needed / O.dt))
    if n_steps > 0:
        O.run(n_steps)

def rotate_mill_by_time(virtual_time_seconds, speed_rpm=9):
    """Run rotation for |virtual_time_seconds| of simulation time. In: seconds, speed_rpm. Out: None."""
    rotation_engine = _get_rotation_engine("rotation_engine")

    rpm = abs(float(speed_rpm))
    omega = rpm * (2*pi) / 60.0  # RADIANES / SEGUNDO

    direction = 1.0 if virtual_time_seconds > 0 else -1.0
    rotation_engine.angularVelocity = direction * omega

    n_steps = int(round(abs(float(virtual_time_seconds)) / O.dt))
    if n_steps > 0:
        O.run(n_steps)

def start_frame_capture(folder_name, interval, runner_label="frameCapture", iter_period=50):
    """Configure periodic CSV sphere dumps via PyRunner. In: folder base, interval (s), labels. Out: runner ref."""
    folder_name = folder_name + str(O.dt) + "_"
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder, exist_ok=True)

    _frameCaptureState.clear()
    _frameCaptureState.update({
        "interval": float(interval),
        "next_save_time": float(O.time),
        "frame_id": 0,
        "folder": folder,
    })

    try:
        r = O.engines[runner_label]
        r.iterPeriod = int(iter_period)
        r.dead = False
    except Exception:
        r = [PyRunner(command="s1_sim_functions._save_sphere_frame()", iterPeriod=int(iter_period), label=runner_label)]
        O.engines += r

    print(f"[FrameCapture] Saving spheres every {interval}s into: {folder}")
    return r


def createBox(x, y, z):
    """Append open-top box facets (half-extents x,y; height z hardcoded 0.375). In: x,y,z dims. Out: None."""
    mat = O.materials.append(FrictMat(density=7850, young=1e9, poisson=0.3, frictionAngle=radians(10)))

    # corner points
    b0 = (-x, -y, 0)
    b1 = ( x, -y, 0)
    b2 = ( x,  y, 0)
    b3 = (-x,  y, 0)
    t0 = (-x, -y, 0.375)
    t1 = ( x, -y, 0.375)
    t2 = ( x,  y, 0.375)
    t3 = (-x,  y, 0.375)

    facets = [
        # bottom (z=0)
        [b0, b1, b2],
        [b0, b2, b3],
        # top (z=0.375)
        [t0, t2, t1],
        [t0, t3, t2],
        # -x wall
        [b0, t0, t3],
        [b0, t3, b3],
        # +x wall
        [b1, b2, t2],
        [b1, t2, t1],
        # -y wall (commented out in your original)
        [b0, b1, t1],
        [b0, t1, t0],
    ]

    for tri in facets:
        O.bodies.append(facet(tri, material=mat))

def createFunnel(x, y, z, fx, fy, dy):
    """Append funnel + deposit box facets (top box x,y,z; deposit fx,fy; funnel length dy). In: dims. Out: None."""
    mat = O.materials.append(FrictMat(density=7850, young=1e9, poisson=0.3, frictionAngle=radians(10)))

    # -- TOP BOX corners -------------------------------------
    # centered at 0,0 so X goes from -x to +x, Y goes from -y/2 to +y/2
    tb0 = (-x, -y/2, 0)
    tb1 = ( x, -y/2, 0)
    tb2 = ( x, -y/2, z)
    tb3 = (-x, -y/2, z)
    tt0 = (-x,  y/2, 0)
    tt1 = ( x,  y/2, 0)
    tt2 = ( x,  y/2, z)
    tt3 = (-x,  y/2, z)

    # -- FUNNEL corners ---------------------------------------
    # top of funnel = bottom of top box at y=-y/2
    # bottom of funnel = y=-y/2-dy, narrowed to fx
    fb0 = (-fx, -y/2-dy, 0)
    fb1 = ( fx, -y/2-dy, 0)
    fb2 = ( fx, -y/2-dy, z)
    fb3 = (-fx, -y/2-dy, z)

    # -- DEPOSIT BOX corners ----------------------------------
    # top = bottom of funnel at y=-y/2-dy
    # bottom = y=-y/2-dy-fy
    db0 = (-fx, -y/2-dy-fy, 0)
    db1 = ( fx, -y/2-dy-fy, 0)
    db2 = ( fx, -y/2-dy-fy, z)
    db3 = (-fx, -y/2-dy-fy, z)

    facets = [
        # -- TOP BOX -----------------------------------------
        # -z wall
        [tb0, tb1, tt1],
        [tb0, tt1, tt0],
        # +z wall
        [tb2, tt2, tt3],
        [tb2, tt3, tb3],
        # -x wall
        [tb0, tt0, tt3],
        [tb0, tt3, tb3],
        # +x wall
        [tb1, tb2, tt2],
        [tb1, tt2, tt1],
        # +y wall (back)
        [tt0, tt1, tt2],
        [tt0, tt2, tt3],

        # -- FUNNEL SLOPES ------------------------------------
        # -x slope
        [tb0, fb0, fb3],
        [tb0, fb3, tb3],
        # +x slope
        [tb1, tb2, fb2],
        [tb1, fb2, fb1],
        # -z slope
        [tb0, tb1, fb1],
        [tb0, fb1, fb0],
        # +z slope
        [tb3, fb3, fb2],
        [tb3, fb2, tb2],

        # -- DEPOSIT BOX --------------------------------------
        # bottom face (closed)
        [db0, db2, db1],
        [db0, db3, db2],
        # -z wall
        [fb0, fb1, db1],
        [fb0, db1, db0],
        # +z wall
        [fb2, fb3, db3],
        [fb2, db3, db2],
        # -x wall
        [fb0, db0, db3],
        [fb0, db3, fb3],
        # +x wall
        [fb1, fb2, db2],
        [fb1, db2, db1],
        # +y wall (back of deposit box)
        [fb3, db3, db2],
        [fb3, db2, fb2],
    ]

    for tri in facets:
        O.bodies.append(facet(tri, material=mat))

def check_overlaps():
    """Max relative sphere–sphere penetration depth. In: O state. Out: float ratio (prints worst pair)."""
    max_rel = 0.0
    worst_pair = None
    for i in O.interactions:
        if not i.isReal:
            continue
        b1 = O.bodies[i.id1]
        b2 = O.bodies[i.id2]
        # skip any contact involving a facet (non-sphere)
        if not isinstance(b1.shape, Sphere) or not isinstance(b2.shape, Sphere):
            continue
        depth = i.geom.penetrationDepth
        r1 = b1.shape.radius
        r2 = b2.shape.radius
        rel = depth / min(r1, r2)
        if rel > max_rel:
            max_rel = rel
            worst_pair = (i.id1, i.id2)
    print(f"Max overlap: {max_rel*100:.3f}% — between bodies {worst_pair}. Depth: {depth}")
    return max_rel

# FUNCIONES AYUDANTES ------------------------------------------------------------------------------------------------||

def _obtain_sag_mill_slice_measurements(sag_mill_body_group):
    """Bounding radius and Z extents from slice body ids. In: id list. Out: (radius_m, z_min, z_max)."""
    x_pos = [O.bodies[mill].state.pos[0] for mill in sag_mill_body_group]
    z_pos = [O.bodies[mill].state.pos[2] for mill in sag_mill_body_group]

    x_min = min(x_pos)
    x_max = max(x_pos)
    z_min = min(z_pos)
    z_max = max(z_pos)

    sag_mill_slice_diameter_m = abs(x_min) + abs(x_max)
    sag_mill_slice_radius_m = abs(x_min)
    sag_mill_slice_depth_m = abs(z_min) + abs(z_max)

    sag_mill_body_dimensions_message = "\nDIAMETER: %s m | RADIUS: %s m | DEPTH: %s m\n" \
    % (sag_mill_slice_diameter_m, sag_mill_slice_radius_m, sag_mill_slice_depth_m)
    print(sag_mill_body_dimensions_message)

    return sag_mill_slice_radius_m, z_min, z_max

def _add_sag_mill_slice_caps(sag_mill_slice_body_group, sag_mill_slice_radius_m, z_min, z_max):
    """Append triangular end-cap facets; extends sag_mill_slice_body_group list. In: group, radius, z. Out: None."""
    cap_max = sag_mill_slice_radius_m * 1.5

    end_cap_back_0 = [Vector3(cap_max,0,z_min), Vector3(0,-cap_max,z_min), Vector3(-cap_max,0,z_min)]
    end_cap_back_1 = [Vector3(cap_max,0,z_min), Vector3(-cap_max,0,z_min), Vector3(0,cap_max,z_min)]

    end_cap_front_0 = [Vector3(cap_max,0,z_max), Vector3(0,-cap_max,z_max), Vector3(-cap_max,0,z_max)]
    end_cap_front_1 = [Vector3(cap_max,0,z_max), Vector3(-cap_max,0,z_max), Vector3(0,cap_max,z_max)]

    sag_mill_slice_body_group += O.bodies.append([facet(end_cap_back_0, color=(1,0,0), wire=True, material="steel")])
    sag_mill_slice_body_group += O.bodies.append([facet(end_cap_back_1, color=(1,0,0), wire=True, material="steel")])
    sag_mill_slice_body_group += O.bodies.append([facet(end_cap_front_0, color=(0,1,0), wire=True, material="steel")])
    sag_mill_slice_body_group += O.bodies.append([facet(end_cap_front_1, color=(0,1,0), wire=True, material="steel")])

def _balance_check():
    """PyRunner hook: stop sim when unbalancedForce below threshold. In: O, BALANCE_STATE. Out: None."""
    unb = unbalancedForce()
    print("UNBALANCED FORCES:", unb)
    BALANCE_STATE["last_unb"] = unb

    if unb < BALANCE_STATE["threshold"]:
        print(f"[balance] BALANCED at iter {O.iter}. UNBALANCED FORCES: {unb:.6g}")
        BALANCE_STATE["done"] = True

        for e in list(O.engines):
            if getattr(e, "label", None) == BALANCE_STATE["label"]:
                O.engines.remove(e)
                break

        O.pause()

def _get_rotation_engine(label="rotation_engine"):
    """Find RotationEngine in O.engines by label. In: label str. Out: engine or raises RuntimeError."""
    try:
        return next(e for e in O.engines if getattr(e, "label", None) == label)
    except StopIteration:
        raise RuntimeError(f"RotationEngine with label='{label}' not found in O.engines")

def _mat_label(b):
    """Material label string for body b, or empty. In: body. Out: str."""
    m = getattr(b, "material", None)
    if m is None:
        return ""
    return str(getattr(m, "label", ""))

def _save_sphere_frame():
    """PyRunner: write sphere CSV when O.time reaches next interval. In: _frameCaptureState. Out: None."""
    print("Checking at time:", O.time)

    st = _frameCaptureState
    if not st:
        return

    t = O.time
    if t < st["next_save_time"]:
        return

    frame_id = st["frame_id"]
    folder = st["folder"]
    filename = os.path.join(folder, f"frame_{frame_id:05d}.csv")

    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","x","y","z","r","m","vx","vy","vz"])

        for b in O.bodies:
            if not b:
                continue
            if not isinstance(getattr(b, "shape", None), Sphere):
                continue

            p = b.state.pos
            v = b.state.vel

            w.writerow([
                int(b.id),
                float(p[0]), float(p[1]), float(p[2]),
                float(b.shape.radius),
                _mat_label(b),
                float(v[0]), float(v[1]), float(v[2]),
            ])

    st["frame_id"] += 1
    st["next_save_time"] += st["interval"]