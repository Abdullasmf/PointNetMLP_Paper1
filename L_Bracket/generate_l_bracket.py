
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path

try:
    import skfem
    from skfem import *
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import dot, grad, div, sym_grad, ddot
except ImportError:
    print("Error: This script requires scikit-fem. Please run: pip install scikit-fem scipy")
    exit(1)

# Check for Gmsh (needed for filleted mesh generation)
try:
    import gmsh
except ImportError:
    print("Error: This script requires gmsh for mesh generation. Please run: pip install gmsh")
    exit(1)

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
num_samples = 5000
lc_fine = 0.005    # Fine mesh size at fillet
lc_coarse = 0.1    # Coarse mesh size away from corner
E_val = 1000.0
nu_val = 0.3
traction_load = np.array([0.0, -15.0]) # Downward split
current_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = Path(current_dir, "L_bracket_stress.h5")

# --- Plane Stress Parameters ---
# Convert E, nu to Lame parameters (mu, lambda)
# Note: For Plane Stress, we modify lambda -> lambda_ps
mu_val = E_val / (2 * (1 + nu_val))
# 3D lambda
lmbda_3d = (E_val * nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
# Plane Stress lambda
lmbda_ps = (2 * mu_val * lmbda_3d) / (lmbda_3d + 2 * mu_val)

# -----------------------------------------------------------------------------
# 2. Physics Definition (scikit-fem)
# -----------------------------------------------------------------------------

# Define the element type: P1 triangles with vector unknowns (displacement u)
# VectorElement combines two scalar elements (u_x, u_y)
e = ElementVector(ElementTriP1())

# Define the Bilinear Form (Stiffness Matrix)
# linear_elasticity helper in skfem handles: lambda * div(u)div(v) + 2*mu * eps(u):eps(v)
# We just need to pass our plane-stress adjusted lambda.
@BilinearForm
def stiffness(u, v, w):
    def epsilon(w):
        return sym_grad(w)
    return lmbda_ps * div(u) * div(v) + 2 * mu_val * ddot(epsilon(u), epsilon(v))

# Define Traction Boundary Condition (Neumann)
# Integration over the boundary 'right'
@LinearForm
def traction_rhs(v, w):
    return dot(w['traction'], v)

# Von Mises Calculation Helper
def compute_von_mises(u_sol, basis):
    # Calculate gradients at quadrature points
    
    # Generate P1 basis for projection
    P1 = ElementTriP1()
    basis_p1 = Basis(basis.mesh, P1)
    
    @BilinearForm
    def mass(u, v, w):
        return u * v
    
    M = asm(mass, basis_p1)
    
    # Define a LinearForm (u, v) -> integral(VM(u) * v)
    @LinearForm
    def rhs_vm(v, w):
        g = grad(w['u'])
        eps_xx = g[0, 0]
        eps_yy = g[1, 1]
        eps_xy = 0.5 * (g[0, 1] + g[1, 0])
        
        # Consistent with Plane Stress
        sig_xx = (lmbda_ps + 2*mu_val)*eps_xx + lmbda_ps*eps_yy
        sig_yy = lmbda_ps*eps_xx + (lmbda_ps + 2*mu_val)*eps_yy
        sig_xy = 2*mu_val*eps_xy
        
        vm_val = np.sqrt(sig_xx**2 - sig_xx*sig_yy + sig_yy**2 + 3*sig_xy**2)
        return vm_val * v

    # To pass 'u' solution into the form, we interpolate it.
    b = asm(rhs_vm, basis_p1, u=basis.interpolate(u_sol))
    
    # Solve M * x = b to get nodal values of Von Mises stress
    vm_nodal = spsolve(M, b)
    return vm_nodal

# -----------------------------------------------------------------------------
# 3. Mesh Generation (Gmsh)
# -----------------------------------------------------------------------------
def generate_l_bracket_mesh(xc_rel, yc_rel, r, lc_fine, lc_coarse,
                             W=1.0, H=1.0, x_offset=0.0, y_offset=0.0):
    """
    Generates a filleted L-bracket mesh using Gmsh.
    Returns a skfem.MeshTri object.

    Parameters
    ----------
    xc_rel, yc_rel : relative position of the re-entrant corner within the
                     bounding box (i.e. distance from x_offset / y_offset).
    r              : fillet radius
    lc_fine        : fine mesh size at fillet
    lc_coarse      : coarse mesh size away from corner
    W, H           : bounding-box width and height
    x_offset, y_offset : bottom-left corner of the bounding box
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0) # Silence
    gmsh.model.add("L_Bracket")

    # Absolute coordinates of the re-entrant corner
    xc = x_offset + xc_rel
    yc = y_offset + yc_rel

    # Tangent points for fillet
    # The sharp corner is at (xc, yc).
    # The fillet connects (xc+r, yc) to (xc, yc+r).
    # Center is at (xc+r, yc+r).

    # Coordinates of points
    p1 = gmsh.model.geo.addPoint(x_offset,         y_offset,         0, lc_coarse)
    p2 = gmsh.model.geo.addPoint(x_offset + W,      y_offset,         0, lc_coarse)
    p3 = gmsh.model.geo.addPoint(x_offset + W,      yc,               0, lc_coarse)

    # Fillet points (High resolution here)
    p4 = gmsh.model.geo.addPoint(xc + r,            yc,               0, lc_fine)
    p5 = gmsh.model.geo.addPoint(xc,                yc + r,           0, lc_fine)
    p_center = gmsh.model.geo.addPoint(xc + r,      yc + r,           0, 0)

    p6 = gmsh.model.geo.addPoint(xc,                y_offset + H,     0, lc_coarse)
    p7 = gmsh.model.geo.addPoint(x_offset,          y_offset + H,     0, lc_coarse)

    # Lines
    l1 = gmsh.model.geo.addLine(p1, p2)      # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)      # Right Lower
    l3 = gmsh.model.geo.addLine(p3, p4)      # Shelf
    
    # Fillet Arc
    # Gmsh usually handles short arc correctly. Directions matter.
    # P4 to P5. Center is (xc+r, yc+r).
    l_fillet = gmsh.model.geo.addCircleArc(p4, p_center, p5)
    
    l4 = gmsh.model.geo.addLine(p5, p6)      # Vertical Inner
    l5 = gmsh.model.geo.addLine(p6, p7)      # Top
    l6 = gmsh.model.geo.addLine(p7, p1)      # Left

    # Surface
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l_fillet, l4, l5, l6])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Extract Elements first to know which nodes are used
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    if len(elemTypes) == 0:
        raise RuntimeError("Gmsh failed to generate 2D elements")
        
    # Flatten all used node tags from triangles
    # elementTypes[0] should be 2 (triangle 3-node).
    # If multiple types, handle them, but here we expect triangles.
    used_node_tags = set()
    tri_node_tags_flat = []
    
    for etype, tags in zip(elemTypes, elemNodeTags):
        if etype == 2: # Triangle
             tri_node_tags_flat.extend(tags)
             for tag in tags:
                 used_node_tags.add(tag)

    # Convert to numpy array for connectivity
    tri_node_tags_flat = np.array(tri_node_tags_flat)
        
    # Extract ALL nodes from Gmsh
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    
    # Filter nodes that are actually used in the mesh
    # Create a mapping from gmsh_tag -> local_index (0..N-1)
    
    # Filter arrays
    used_node_tags_list = sorted(list(used_node_tags))
    gmsh_tag_to_skfem_idx = {tag: i for i, tag in enumerate(used_node_tags_list)}
    
    # Construct points array (2, N_used_nodes)
    # We need to find coordinates for each used tag.
    # gmsh 'nodeTags' and 'coords' correspond.
    # Build a lookup for coords.
    tag_to_coords = {}
    
    # coords is flat [x,y,z, x,y,z...]
    # reshape to (N_all, 3)
    coords_reshaped = np.array(coords).reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        tag_to_coords[tag] = coords_reshaped[i, :2] # Keep x,y
        
    # Now build the final cleaned points array
    clean_points = []
    for tag in used_node_tags_list:
        clean_points.append(tag_to_coords[tag])
    
    clean_points = np.array(clean_points).T # (2, N_nodes)
    
    # Rebuild connectivity with new indices
    # (N_tri, 3) -> Transpose to (3, N_tri)
    new_triangles = []
    # Process the flat list in chunks of 3
    for k in range(0, len(tri_node_tags_flat), 3):
        t1 = tri_node_tags_flat[k]
        t2 = tri_node_tags_flat[k+1]
        t3 = tri_node_tags_flat[k+2]
        new_triangles.append([
            gmsh_tag_to_skfem_idx[t1],
            gmsh_tag_to_skfem_idx[t2],
            gmsh_tag_to_skfem_idx[t3]
        ])
        
    new_triangles = np.array(new_triangles).T

    gmsh.finalize()
    
    return MeshTri(clean_points, new_triangles)


# -----------------------------------------------------------------------------
# 4. Main Loop
# -----------------------------------------------------------------------------

print(f"Generating {num_samples} samples using scikit-fem + Gmsh...")

with h5py.File(output_filename, 'w') as hf:
    for i in range(num_samples):
        # --- A. Generate variable L-bracket Mesh ---
        # Per-sample fillet radius: uniform between sharp (0.01) and smooth (0.1)
        fillet_radius = np.random.uniform(0.02, 0.1)

        # Asymmetric bounding box parameters
        W        = np.random.uniform(0.8, 1.2)
        H        = np.random.uniform(0.8, 1.2)
        x_offset = np.random.uniform(-0.5, 0.5)
        y_offset = np.random.uniform(-0.5, 0.5)

        # Relative corner position within the bounding box.
        # Enforce minimum leg thickness of 0.2 on both sides.
        xc_rel = np.random.uniform(0.2, W - 0.2)
        yc_rel = np.random.uniform(0.2, H - 0.2)

        # Absolute corner coordinates (used for BCs and metadata)
        xc = x_offset + xc_rel
        yc = y_offset + yc_rel

        # Use helper for filleted mesh
        mesh = generate_l_bracket_mesh(xc_rel, yc_rel, fillet_radius,
                                        lc_fine, lc_coarse,
                                        W, H, x_offset, y_offset)
            
        # --- B. Solver Setup ---
        basis = Basis(mesh, e)
        
        # Boundaries
        dofs = basis.get_dofs()
        
        # 1. Clamp Top (Dirichlet): y == y_offset + H
        top_dofs = basis.get_dofs(
            lambda x, _yo=y_offset, _H=H: np.isclose(x[1], _yo + _H, atol=1e-3)
        )
        D = top_dofs.all()
        
        # 2. Traction Right (Neumann): x == x_offset + W
        right_facets = mesh.facets_satisfying(
            lambda x, _xo=x_offset, _W=W: np.isclose(x[0], _xo + _W)
        )
        
        # Create a specific basis restricted to the boundary facets for integration
        basis_right = FacetBasis(mesh, e, facets=right_facets)
        
        # Assemble matrices
        K = asm(stiffness, basis)
        
        # Prepare traction field (interpolate constant vector to FacetBasis)
        # --- Dynamic Traction Scaling (Dual-Leg Bending Protection) ---
        # Bending stress scales proportionally to Lever_Arm / Thickness^2.
        # We calculate this vulnerability factor for BOTH legs.
        
        # 1. Horizontal Leg Vulnerability: Lever arm is roughly (W - xc_rel). Thickness is yc_rel.
        S_h = (W - xc_rel) / (yc_rel ** 2)
        
        # 2. Vertical Leg Vulnerability: Lever arm is roughly (W - xc_rel/2). Thickness is xc_rel.
        S_v = (W - (xc_rel / 2)) / (xc_rel ** 2)
        
        # Find the weakest geometric link
        S_max = max(S_h, S_v)
        
        # In the baseline geometry (W=1, xc=0.5, yc=0.5), S_max is 3.0.
        # Baseline total force is -7.5 (traction -15.0 * boundary length 0.5).
        # Target constant = Force * S_max = -7.5 * 3.0 = -22.5
        target_moment_capacity = -22.5
        
        # Calculate the maximum safe total force for this specific randomized geometry
        safe_total_force = target_moment_capacity / S_max
        
        # Convert total force back to distributed traction (Force / boundary length)
        traction_val = safe_total_force / yc_rel
        
        x_quad = basis_right.global_coordinates().value 
        
        traction_data = np.zeros_like(x_quad)
        traction_data[1, :, :] = traction_val # Set y-component
        
        traction_field = DiscreteField(value=traction_data)
        
        f = asm(traction_rhs, basis_right, traction=traction_field)
        
        # Enforce Dirichlet BCs
        u_sol = solve(*condense(K, f, D=D))
        
        # --- C. Post-Process (Von Mises) ---
        vm_stress = compute_von_mises(u_sol, basis)
        
        # --- D. Save ---
        points = mesh.p.T # (N_nodes, 2)
        
        grp = hf.create_group(f"sample_{i}")
        grp.create_dataset("points", data=points)
        grp.create_dataset("stress", data=vm_stress.reshape(-1, 1))
        # Expanded corner: [xc, yc, W, H, x_offset, y_offset, fillet_radius]
        # corner[0:2] = absolute corner position (backward compatible)
        grp.create_dataset("corner", data=np.array(
            [xc, yc, W, H, x_offset, y_offset, fillet_radius]
        ))
        
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{num_samples}")

    print(f"Saved {num_samples} samples to {output_filename}")

# -----------------------------------------------------------------------------
# 4. Verification Check
# -----------------------------------------------------------------------------
print("Verifying last sample...")
plt.figure(figsize=(10, 8))
# Skfem mesh is easily plotting with matplotlib using triangulation
import matplotlib.tri as tri

# The 'points' and 'vm_stress' from the last loop iteration are still in scope
triangulation = tri.Triangulation(points[:, 0], points[:, 1], mesh.t.T)

plt.tricontourf(triangulation, vm_stress, levels=40, cmap='jet')
plt.colorbar(label="Von Mises Stress")
# Overlay the mesh
plt.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.3)
#plt.plot(xc, yc, 'wo', markeredgecolor='k', markersize=10, label="Re-entrant Corner")
plt.title(f"Sample {num_samples}: Variable L-Bracket\nCorner ({xc:.2f}, {yc:.2f})")
plt.axis('equal')
# plt.legend()
plt.show()

