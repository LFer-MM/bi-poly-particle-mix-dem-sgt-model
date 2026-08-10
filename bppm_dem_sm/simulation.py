"""Random-mix ingress simulation for the bidisperse SAG mill slice (YADE)."""

from yade import qt

from . import config, sim_functions


def run():
    """Set up the mill and ingress a random bidisperse particle charge.

    Initializes YADE materials, loads the SAG mill STL slice, configures
    Hertz–Mindlin contacts, opens a Qt viewer, and calls
    :func:`sim_functions.ingress_random` with the bidisperse rock/steel charge.

    Returns:
        None
    """
    sim_functions.initialize_simulation_materials(config.MATERIALS)
    sim_functions.initialize_sag_mill_slice(config.SAGMILL_STL_PATH)
    sim_functions.initialize_engines(
        contact_model="hertz_mindlin",
        contact_model_params=config.build_material_interactions(),
        rotation_engine=False,
    )
    sim_functions.set_dt(new_dt=0.000015 * 0.2)
    # s1_sim_functions.load_particle_positions("rmic_nopf_settled.csv")
    sim_functions.set_gravity_damping(new_gravity_damping=0.2)

    qt.Controller()
    qt.View()

    sim_functions.ingress_random(
        diameter=11.0,
        depth=0.375,
        r_small=0.034925,
        r_large=0.06985,
        n_small=994,
        n_large=234,
        box_height=1.0,
        material_small="rock",
        material_large="steel",
        color_small=(1, 0, 0),
        color_large=(0, 0, 1),
        settle_steps=10000000,
        padding=0.1,
    )

# s1_sim_functions.set_gravity_damping(new_gravity_damping=0.0)

# s1_sim_functions.check_overlaps()

# ingress_func_v1.get_particle_inventory(0.06985/2, 0.1397/2)

# s1_sim_functions.load_ball_particles(s0_global_sim_config.ball_diam_m, s0_global_sim_config.ball_count)

# EJECUTAR MANUALMENTE EN LA TERMINAL DE YADE -> s1_sim_functions.settle_balance_save(0.2, "rmic_nopf_settled.csv")

# s1_sim_functions.run_until_forces_balanced(threshold=0.01)

# s1_sim_functions.save_particle_positions("rmic_nopf.csv")


if __name__ == "__main__":
    run()
