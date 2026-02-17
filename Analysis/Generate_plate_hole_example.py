import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import os

try:
    import skfem
    from skfem import *
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import dot, grad, div, sym_grad, ddot
except ImportError:
    print("Error: This script requires scikit-fem. Please run: pip install scikit-fem scipy")
    exit(1)

# Check for Gmsh (needed for mesh generation)
try:
    import gmsh
except ImportError:
    print("Error: This script requires gmsh for mesh generation. Please run: pip install gmsh")
    exit(1)

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
num_samples = 1
lc_fine = 0.015    # Fine mesh size at hole boundary
lc_coarse = 0.1    # Coarse mesh size away from hole
E_val = 1000.0
nu_val = 0.3
traction_load = np.array([15.0, 0.0]) # Tension in X direction
current_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(current_dir, "Plate_hole_example.h5")

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
e = ElementVector(ElementTriP1())

# Define the Bilinear Form (Stiffness Matrix)
@BilinearForm
def stiffness(u, v, w):
    def epsilon(w):
        return sym_grad(w)
    return lmbda_ps * div(u) * div(v) + 2 * mu_val * ddot(epsilon(u), epsilon(v))

# Define Traction Boundary Condition (Neumann)
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
def generate_plate_hole_mesh(cx, cy, r, lc_fine, lc_coarse):
    """
    Generates a 1x1 plate with a circular hole using Gmsh.
    Returns a skfem.MeshTri object.
    
    cx, cy: Center of the hole
    r: Radius of the hole
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0) # Silence
    gmsh.model.add("Plate_Hole")

    # Dimensions: 1x1 Square
    
    # 1. Outer Square Points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc_coarse)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc_coarse)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, lc_coarse)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc_coarse)

    l1 = gmsh.model.geo.addLine(p1, p2)      # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)      # Right
    l3 = gmsh.model.geo.addLine(p3, p4)      # Top
    l4 = gmsh.model.geo.addLine(p4, p1)      # Left

    loop_square = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # 2. Inner Hole Points
    # Defined counter-clockwise equivalent or handled by PlaneSurface substraction
    pc = gmsh.model.geo.addPoint(cx, cy, 0, lc_fine) # Center helper

    ph1 = gmsh.model.geo.addPoint(cx + r, cy, 0, lc_fine)
    ph2 = gmsh.model.geo.addPoint(cx, cy + r, 0, lc_fine)
    ph3 = gmsh.model.geo.addPoint(cx - r, cy, 0, lc_fine)
    ph4 = gmsh.model.geo.addPoint(cx, cy - r, 0, lc_fine)

    c1 = gmsh.model.geo.addCircleArc(ph1, pc, ph2)
    c2 = gmsh.model.geo.addCircleArc(ph2, pc, ph3)
    c3 = gmsh.model.geo.addCircleArc(ph3, pc, ph4)
    c4 = gmsh.model.geo.addCircleArc(ph4, pc, ph1)

    loop_hole = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])

    # 3. Surface with hole
    # First loop is outer, subsequent loops are holes
    surf = gmsh.model.geo.addPlaneSurface([loop_square, loop_hole])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # --- Extract Elements (Manually to ensure clean skfem import) ---
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    if len(elemTypes) == 0:
        raise RuntimeError("Gmsh failed to generate 2D elements")
        
    used_node_tags = set()
    tri_node_tags_flat = []
    
    # Flatten all used node tags from triangles (type 2)
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
    used_node_tags_list = sorted(list(used_node_tags))
    gmsh_tag_to_skfem_idx = {tag: i for i, tag in enumerate(used_node_tags_list)}
    
    # Construct points array (2, N_used_nodes)
    tag_to_coords = {}
    coords_reshaped = np.array(coords).reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        tag_to_coords[tag] = coords_reshaped[i, :2] # x, y
        
    clean_points = []
    for tag in used_node_tags_list:
        clean_points.append(tag_to_coords[tag])
    
    clean_points = np.array(clean_points).T # (2, N_nodes)
    
    # Rebuild connectivity
    new_triangles = []
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
        # --- A. Generate variable Plate Hole Mesh ---
        # Center varies from 0.3 to 0.7
        cx = np.random.uniform(0.3, 0.7)
        cy = np.random.uniform(0.3, 0.7)
        # Radius varies from 0.1 to 0.2
        r_hole = np.random.uniform(0.1, 0.2)
        
        # Guard against hole hitting boundary (margin: center +/- radius should be within [0,1])
        # 0.3 - 0.2 = 0.1 (safe)
        # 0.7 + 0.2 = 0.9 (safe)
        
        mesh = generate_plate_hole_mesh(cx, cy, r_hole, lc_fine, lc_coarse)
            
        # --- B. Solver Setup ---
        basis = Basis(mesh, e)
        
        # Boundaries:
        # Left (x=0): Clamp (Dirichlet)
        # Right (x=1): Traction (Neumann)
        
        left_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 0.0, atol=1e-3))
        D = left_dofs.all() # Clamp all components on left edge
        
        right_facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
        basis_right = FacetBasis(mesh, e, facets=right_facets)
        
        # Assemble Stiffness
        K = asm(stiffness, basis)
        
        # Prepare traction field (Horizontal pull)
        # Load is (10.0, 0.0) corresponding to traction_load
        trax_val_x = float(traction_load[0])
        trax_val_y = float(traction_load[1])
        
        x_quad = basis_right.global_coordinates().value 
        traction_data = np.zeros_like(x_quad)
        traction_data[0, :, :] = trax_val_x 
        traction_data[1, :, :] = trax_val_y
        
        traction_field = DiscreteField(value=traction_data)
        
        f = asm(traction_rhs, basis_right, traction=traction_field)
        
        # Solve
        u_sol = solve(*condense(K, f, D=D))
        
        # --- C. Post-Process (Von Mises) ---
        vm_stress = compute_von_mises(u_sol, basis)
        
        # --- D. Save ---
        points = mesh.p.T # (N_nodes, 2)
        
        grp = hf.create_group(f"sample_{i}")
        grp.create_dataset("points", data=points)
        grp.create_dataset("stress", data=vm_stress.reshape(-1, 1))
        # Save geometry parameters: center_x, center_y, radius
        grp.create_dataset("params", data=np.array([cx, cy, r_hole]))
        
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{num_samples}")

    print(f"Saved {num_samples} samples to {output_filename}")

# -----------------------------------------------------------------------------
# 5. Verification Check
# -----------------------------------------------------------------------------
print("Verifying last sample...")
plt.figure(figsize=(10, 8))
import matplotlib.tri as tri

triangulation = tri.Triangulation(points[:, 0], points[:, 1], mesh.t.T)

plt.tricontourf(triangulation, vm_stress, levels=40, cmap='jet')
plt.colorbar(label="Von Mises Stress")
plt.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.3)

# Overlay hole outline
theta = np.linspace(0, 2*np.pi, 100)
x_circ = cx + r_hole * np.cos(theta)
y_circ = cy + r_hole * np.sin(theta)
plt.plot(x_circ, y_circ, 'w-', linewidth=2)

plt.title(f"Sample {num_samples}: Plate with Hole\nCenter ({cx:.2f}, {cy:.2f}), R={r_hole:.2f}")
plt.axis('equal')
plt.show()
