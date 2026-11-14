import jax
import jax.numpy as jnp


def hyperrectangles_isdisjoint(lb1, ub1, lb2, ub2):
    r"""
    Check if two hyperrectangles are disjoint.

    Algorithm:
    ``H1 \cap H2 are not disjoint iff |c_2 - c_1| \leq r_1 + r_2, where \leq is taken component-wise.``
    """
    # Compute both centers
    center1 = (ub1 + lb1) / 2
    center2 = (ub2 + lb2) / 2

    # Compute both radii
    radius1 = (ub1 - lb1) / 2
    radius2 = (ub2 - lb2) / 2
    center_diff = center2 - center1
    empty_intersection = jnp.any(jnp.abs(center_diff) > radius1 + radius2)

    return empty_intersection


vmap_hyperrectangles_isdisjoint = jax.jit(jax.vmap(hyperrectangles_isdisjoint, in_axes=(None, None, 0, 0), out_axes=0))


def hyperrectangles_iscontained(lb1, ub1, lb2, ub2):
    """
    Check if hyperrectangle (lb1, ub1) is fully contained in hyperrectangle (lb2, ub2).

    Algorithm:
    ``H1 is contained in H2 iff lb2 <= lb1 and ub1 <= ub2, where <= is taken component-wise.``
    """
    return jnp.all((lb1 >= lb2) & (ub1 <= ub2))


vmap_hyperrectangles_iscontained = jax.jit(jax.vmap(hyperrectangles_iscontained, in_axes=(None, None, 0, 0), out_axes=0))


def hyperrectangles_iscontained_multi(lb1, ub1, lbs2, ubs2):
    """
    Check if one hyperrectangle (lb1, ub1) is contained in any of multiple other hyperrectangles
    given by (lbs2, ubs2).
    """
    contained = vmap_hyperrectangles_iscontained(lb1, ub1, lbs2, ubs2)

    return jnp.any(contained)


def hyperrectangles_isdisjoint_multi(lb1, ub1, lbs2, ubs2):
    r"""
    Check if one hyperrectangle (lb1, ub1) is disjoint from multiple other hyperrectangles given by (lbs2, ubs2)

    Algorithm:
    ``H1 \cap H2 are not disjoint iff |c_2 - c_1| \leq r_1 + r_2, where \leq is taken component-wise.``
    """
    empty_intersections = vmap_hyperrectangles_isdisjoint(lb1, ub1, lbs2, ubs2)
    empty_intersection = jnp.all(empty_intersections)

    return empty_intersection
