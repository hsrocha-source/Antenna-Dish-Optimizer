import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib.patches as patches



D = 15.0
X_OFFSET = 0.0
Y_OFFSET = 0.0

def reflector_surface(x, y, params, F):

    R_max = D / 2.0

    # Base Parabola (relative to global origin)
    z_base = (x**2 + y**2) / (4.0 * F) - F

    # Zernike Local Coordinates
    x_local = x - X_OFFSET
    y_local = y - Y_OFFSET
    r_local = jnp.sqrt(x_local**2 + y_local**2)
    rho = jnp.clip(r_local / R_max, 0.0, 1.0)

    phi_az = jnp.arctan2(y_local, x_local)

    # Zernike Polynomials (Noll Indexing Z4 to Z22) stored safely for JAX
    # Note these are quasi-ortogonal here. only ortogonal in unit circle
    Z = {}
    Z[4]  = jnp.sqrt(3.0) * (2.0*rho**2 - 1.0)
    Z[5]  = jnp.sqrt(6.0) * (rho**2) * jnp.sin(2.0*phi_az)
    Z[6]  = jnp.sqrt(6.0) * (rho**2) * jnp.cos(2.0*phi_az)
    Z[7]  = jnp.sqrt(8.0) * (3.0*rho**3 - 2.0*rho) * jnp.sin(phi_az)
    Z[8]  = jnp.sqrt(8.0) * (3.0*rho**3 - 2.0*rho) * jnp.cos(phi_az)
    Z[9]  = jnp.sqrt(8.0) * (rho**3) * jnp.sin(3.0*phi_az)
    Z[10] = jnp.sqrt(8.0) * (rho**3) * jnp.cos(3.0*phi_az)
    Z[11] = jnp.sqrt(5.0) * (6.0*rho**4 - 6.0*rho**2 + 1.0)
    Z[12] = jnp.sqrt(10.0) * (4.0*rho**4 - 3.0*rho**2) * jnp.cos(2.0*phi_az)
    Z[13] = jnp.sqrt(10.0) * (4.0*rho**4 - 3.0*rho**2) * jnp.sin(2.0*phi_az)
    Z[14] = jnp.sqrt(10.0) * (rho**4) * jnp.cos(4.0*phi_az)
    Z[15] = jnp.sqrt(10.0) * (rho**4) * jnp.sin(4.0*phi_az)
    Z[16] = jnp.sqrt(12.0) * (10.0*rho**5 - 12.0*rho**3 + 3.0*rho) * jnp.cos(phi_az)
    Z[17] = jnp.sqrt(12.0) * (10.0*rho**5 - 12.0*rho**3 + 3.0*rho) * jnp.sin(phi_az)
    Z[18] = jnp.sqrt(12.0) * (5.0*rho**5 - 4.0*rho**3) * jnp.cos(3.0*phi_az)
    Z[19] = jnp.sqrt(12.0) * (5.0*rho**5 - 4.0*rho**3) * jnp.sin(3.0*phi_az)
    Z[20] = jnp.sqrt(12.0) * (rho**5) * jnp.cos(5.0*phi_az)
    Z[21] = jnp.sqrt(12.0) * (rho**5) * jnp.sin(5.0*phi_az)
    Z[22] = jnp.sqrt(7.0) * (20.0*rho**6 - 30.0*rho**4 + 12.0*rho**2 - 1.0)

    z_zernike = jnp.sum(jnp.array([params[f'z{i}'] * Z[i] for i in range(4, 23)]))

    return z_base + z_zernike

def get_normal(x, y, params, F):
    df_dx = jax.grad(reflector_surface, argnums=0)(x, y, params, F)
    df_dy = jax.grad(reflector_surface, argnums=1)(x, y, params, F)
    n = jnp.array([-df_dx, -df_dy, 1.0])
    return n / jnp.linalg.norm(n)


def trace_ray(x, y, theta_az_deg, theta_el_deg, params, F, sharpness):
    # U, V Direction Cosines (Native DBF steering)
    az = theta_az_deg * jnp.pi / 180.0
    el = theta_el_deg * jnp.pi / 180.0

    z_surf = reflector_surface(x, y, params, F)
    p_surf = jnp.array([x, y, z_surf])

    u = jnp.sin(az)
    v = jnp.sin(el)
    w = -jnp.sqrt(jnp.maximum(1.0 - u**2 - v**2, 1e-8)) # Protect sqrt for JAX
    
    d_in = jnp.array([u, v, w])

    n_surf = get_normal(x, y, params, F)
    d_out = d_in - 2.0 * jnp.dot(d_in, n_surf) * n_surf

    # Tilted Array Face Geometry (Sits at the Origin / Focal Plane)
    z_center = reflector_surface(X_OFFSET, Y_OFFSET, params, F)
    v_chief = jnp.array([-X_OFFSET, -Y_OFFSET, -z_center])
    n_arr = v_chief / jnp.linalg.norm(v_chief)

    u_temp = jnp.array([1.0, 0.0, 0.0])
    u_vec = u_temp - jnp.dot(u_temp, n_arr) * n_arr
    u_vec = u_vec / jnp.linalg.norm(u_vec)
    v_vec = jnp.cross(n_arr, u_vec)

    # DYNAMIC SHADOW CALCULATION
    p_surf_static = jax.lax.stop_gradient(p_surf)
    t_in = jnp.dot(p_surf_static, n_arr) / (jnp.dot(d_in, n_arr) + 1e-8)
    p_block = p_surf_static - t_in * d_in

    x_block_local = jnp.dot(p_block, u_vec)
    y_block_local = jnp.dot(p_block, v_vec)


    # Approaches 1.0 when inside the block region, 0.0 outside
    soft_blocked_x = jax.nn.sigmoid(sharpness * (0.5 - jnp.abs(x_block_local)))
    soft_blocked_y = jax.nn.sigmoid(sharpness * (2.5 - jnp.abs(y_block_local)))
    soft_is_blocked = soft_blocked_x * soft_blocked_y

    t_out = -jnp.dot(p_surf, n_arr) / (jnp.dot(d_out, n_arr) + 1e-8)
    p_hit = p_surf + t_out * d_out

    # Approaches 1.0 when t_out > 0, 0.0 when t_out < 0
    soft_valid_reflection = jax.nn.sigmoid(sharpness * t_out)

    # valid_mask = 1.0 when NOT blocked AND valid reflection
    valid_mask = (1.0 - soft_is_blocked) * soft_valid_reflection

    path_length = jnp.dot(p_surf, d_in) + t_out

    x_local = jnp.dot(p_hit, u_vec)
    y_local = jnp.dot(p_hit, v_vec)

    return x_local, y_local, path_length, valid_mask

trace_rays_vmap = jax.vmap(trace_ray, in_axes=(0, 0, 0, 0, None, None, None))

def evaluate_wavefront(params, X_2d, Y_2d, az_grid, el_grid, F, aperture_mask_2d):
    # Extract physical step sizes directly from the grid
    dx = jnp.abs(X_2d[0, 1] - X_2d[0, 0])
    dy = jnp.abs(Y_2d[1, 0] - Y_2d[0, 0])

    def eval_angle(az, el):
        _, _, path_lengths_2d, valid_mask_2d = trace_rays_vmap(
            X_2d.flatten(), Y_2d.flatten(), 
            jnp.full_like(X_2d.flatten(), az), jnp.full_like(Y_2d.flatten(), el), 
            params, F, 500
        )
        
        path_lengths = path_lengths_2d.reshape(X_2d.shape)
        valid_mask = valid_mask_2d.reshape(X_2d.shape)
        
        grad_x = jnp.abs(jnp.diff(path_lengths, axis=1)) / dx 
        grad_y = jnp.abs(jnp.diff(path_lengths, axis=0)) / dy 
        
        valid_x = valid_mask[:, :-1] * valid_mask[:, 1:] * aperture_mask_2d[:, :-1] * aperture_mask_2d[:, 1:]
        valid_y = valid_mask[:-1, :] * valid_mask[1:, :] * aperture_mask_2d[:-1, :] * aperture_mask_2d[1:, :]
        
        max_grad_x = jnp.max(grad_x * valid_x)
        max_grad_y = jnp.max(grad_y * valid_y)
        
        return jnp.maximum(max_grad_x, max_grad_y)
        
    eval_angles_vmap = jax.vmap(eval_angle, in_axes=(0, 0))
    return jnp.max(eval_angles_vmap(az_grid, el_grid))

def evaluate_array_aliasing(params, X_2d, Y_2d, az_grid, el_grid, F, aperture_mask_2d, element_spacing=0.11):
    # element_spacing of 0.11m assumes lambda/2 spacing for L-band (0.22m)
    
    def eval_angle(az, el):
        # Trace rays from the structured aperture grid
        x_hit_flat, y_hit_flat, path_lengths_flat, valid_mask_flat = trace_rays_vmap(
            X_2d.flatten(), Y_2d.flatten(), 
            jnp.full_like(X_2d.flatten(), az), jnp.full_like(Y_2d.flatten(), el), 
            params, F, 500
        )
        
        # Reshape back to 2D to preserve adjacency
        x_hit = x_hit_flat.reshape(X_2d.shape)
        y_hit = y_hit_flat.reshape(X_2d.shape)
        path_lengths = path_lengths_flat.reshape(X_2d.shape)
        valid_mask = valid_mask_flat.reshape(X_2d.shape)
        
        # Mask: Valid only if BOTH adjacent rays hit the array cleanly
        valid_x = valid_mask[:, :-1] * valid_mask[:, 1:] * aperture_mask_2d[:, :-1] * aperture_mask_2d[:, 1:]
        valid_y = valid_mask[:-1, :] * valid_mask[1:, :] * aperture_mask_2d[:-1, :] * aperture_mask_2d[1:, :]
        
        # --- Gradients in the Aperture X-direction mapped to Array ---
        dx_hit_x = x_hit[:, 1:] - x_hit[:, :-1]
        dy_hit_x = y_hit[:, 1:] - y_hit[:, :-1]
        ds_array_x = jnp.sqrt(dx_hit_x**2 + dy_hit_x**2) + 1e-8 # Physical distance on array
        dl_x = jnp.abs(path_lengths[:, 1:] - path_lengths[:, :-1]) # Path length diff
        
        grad_array_x = (dl_x / ds_array_x) * valid_x
        
        # --- Gradients in the Aperture Y-direction mapped to Array ---
        dx_hit_y = x_hit[1:, :] - x_hit[:-1, :]
        dy_hit_y = y_hit[1:, :] - y_hit[:-1, :]
        ds_array_y = jnp.sqrt(dx_hit_y**2 + dy_hit_y**2) + 1e-8 # Physical distance on array
        dl_y = jnp.abs(path_lengths[1:, :] - path_lengths[:-1, :]) # Path length diff
        
        grad_array_y = (dl_y / ds_array_y) * valid_y
        
        # Get the maximum spatial gradient (meters of path difference per meter of array)
        max_grad = jnp.maximum(jnp.max(grad_array_x), jnp.max(grad_array_y))
        
        # Multiply by element spacing to get the max path difference between adjacent DBF elements
        return max_grad * element_spacing
        
    eval_angles_vmap = jax.vmap(eval_angle, in_axes=(0, 0))
    # Return the worst-case element-to-element path difference across all tested angles
    return jnp.max(eval_angles_vmap(az_grid, el_grid))

def loss_fn(params, x_aperture, y_aperture, az_grid, el_grid, F, sharpness):
    def simulate_angle(az, el):
        x_hit, y_hit, path_length, valid_mask = trace_rays_vmap(
            x_aperture, y_aperture,
            jnp.full_like(x_aperture, az), jnp.full_like(y_aperture, el),
            params, F, sharpness
        )

        num_valid = jnp.sum(valid_mask) + 1e-8

        spill_x = jax.nn.softplus(sharpness * (jnp.abs(x_hit) - 0.5)) / sharpness
        spill_y = jax.nn.softplus(sharpness * (jnp.abs(y_hit) - 2.5)) / sharpness
        spill_raw = jnp.square(spill_x) + jnp.square(spill_y)
        
        spill = jnp.sum(spill_raw * valid_mask) / num_valid

        # We define where the beam *should* roughly be.
        x_target_ideal = -(az / 4.0) * 0.35
        y_target_ideal = -(el / 15.0) * 1.8

        mean_x = jnp.sum(x_hit * valid_mask) / num_valid
        mean_y = jnp.sum(y_hit * valid_mask) / num_valid

        # Allow the centroid to settle anywhere within a natural margin
        # e.g., ±0.08m in X and ±0.2m in Y before triggering a penalty
        margin_x = 0.08
        margin_y = 0.20

        dx = jnp.abs(mean_x - x_target_ideal)
        dy = jnp.abs(mean_y - y_target_ideal)

        centroid_loss = jnp.square(jax.nn.relu(dx - margin_x)) + jnp.square(jax.nn.relu(dy - margin_y))

        var_x = jnp.sum(jnp.square(x_hit - mean_x) * valid_mask) / num_valid
        var_y = jnp.sum(jnp.square(y_hit - mean_y) * valid_mask) / num_valid
        spread_loss = var_x + var_y
        lost_ray_penalty = jnp.mean(1.0 - valid_mask)

        mean_path = jnp.sum(path_length * valid_mask) / num_valid
        path_variance = jnp.sum(jnp.square(path_length - mean_path) * valid_mask) / num_valid

        return spill, centroid_loss, spread_loss, lost_ray_penalty, path_variance

    simulate_angles_vmap = jax.vmap(simulate_angle, in_axes=(0, 0))
    spills, centroid_losses, spread_losses, lost_ray_penalty, path_variance = simulate_angles_vmap(az_grid, el_grid)

    z_losses = [jnp.square(params[f'z{i}']) * (1.0 if i < 9 else (5.0 if i < 16 else 20.0)) for i in range(4, 23)]
    reg_loss = jnp.sum(jnp.array(z_losses))


      # Weight as needed
    return (2.0 * jnp.mean(spills)) + (0.1 * reg_loss) + jnp.mean(lost_ray_penalty) * 50.0, jnp.max(path_variance)

def optimize_reflector(F, x_aperture, y_aperture, X_2d, Y_2d, aperture_mask_2d):
    initial_params = {}
    for i in range(4, 23):
        initial_params[f'z{i}'] = 0.0


    # Angular sampling
    az_lin = jnp.linspace(-4.0, 4.0, 9)
    el_lin = jnp.linspace(-15.0, 15.0, 31)
    AZ, EL = jnp.meshgrid(az_lin, el_lin)
    az_grid = AZ.flatten()
    el_grid = EL.flatten()

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(initial_params)

    @jax.jit
    def step(params, opt_state, F_val, current_sharpness):
        (loss, max_path_var), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_aperture, y_aperture, az_grid, el_grid, F_val, current_sharpness
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return loss, new_params, opt_state, max_path_var

    params = initial_params
    epochs = 5000
    lambda_L_band = 0.22

    print(f"Starting L-Band GO+DBF optimization for a 15m dish...")
    for i in range(epochs):
        current_sharpness = 5.0 + (495.0 * (i / epochs))

        loss, params, opt_state, _ = step(params, opt_state, F, current_sharpness)

        if i % 1500 == 0:
            # 1. Evaluate traditional wavefront gradient on the aperture
            max_aperture_gradient = evaluate_wavefront(params, X_2d, Y_2d, az_grid, el_grid, F, aperture_mask_2d)
            
            # 2. Evaluate element-to-element aliasing on the array (Assuming 0.11m spacing for L-band)
            element_spacing = 0.11
            max_element_diff = evaluate_array_aliasing(params, X_2d, Y_2d, az_grid, el_grid, F, aperture_mask_2d, element_spacing)
            
            grad_in_lambdas_per_m = max_aperture_gradient / lambda_L_band
            diff_in_lambdas = max_element_diff / lambda_L_band
            
            print(f"Epoch {i:5d} | Loss: {loss:.6f} | Focus (f/D): {F/D:.6f}")
            print(f"          -> Max Aperture Gradient: {max_aperture_gradient:.4f} m/m ({grad_in_lambdas_per_m:.2f} λ/m)")
            print(f"          -> Max Adjacent Element $\Delta$L: {max_element_diff:.4f} m ({diff_in_lambdas:.2f} λ)")
            
            # Threshold Check: If Delta L > lambda/2, DBF will alias
            if diff_in_lambdas > 0.5:
                print("          -> [CRITICAL WARNING]: Element-to-element phase difference > 0.5 λ. DBF spatial aliasing will occur!")
            elif jnp.max(max_element_diff) > 0.0: 
                # Note: The caustic check
                # If rays cross over each other, ds_array approaches 0, and the gradient explodes.
                if max_element_diff > 10.0:
                    print("          -> [CRITICAL WARNING]: Caustic detected! Rays are crossing before hitting the array.")
    return params


def plot_ray_footprints(params, F, D=15.0):
    print("Generating focal plane ray trace plots...")

    num_spatial = 50
    x_lin = jnp.linspace(X_OFFSET - D/2, X_OFFSET + D/2, num_spatial)
    y_lin = jnp.linspace(Y_OFFSET - D/2, Y_OFFSET + D/2, num_spatial)
    X, Y = jnp.meshgrid(x_lin, y_lin)

    aperture_mask = ((X - X_OFFSET)**2 + (Y - Y_OFFSET)**2) <= (D/2)**2
    valid_rays_mask = aperture_mask 

    x_aperture = X[valid_rays_mask].flatten()
    y_aperture = Y[valid_rays_mask].flatten()

    test_angles = [
        (0.0, 0.0),      # Boresight
        (4.0, 15.0),     # Top Right Extreme
        (-4.0, 15.0),    # Top Left Extreme
        (4.0, -15.0),    # Bottom Right Extreme
        (-4.0, -15.0),   # Bottom Left Extreme
        (4.0, 0.0),      # Right Edge
        (0.0, 15.0)      # Top Edge
    ]

    fig, ax = plt.subplots(figsize=(6, 12))

    # Array is centered at (0,0) in local coordinates.
    # X bounds: [-0.5, 0.5], Y bounds: [-2.5, 2.5]
    array_rect = patches.Rectangle((-0.5, -2.5), 1.0, 5.0, linewidth=2,
                                   edgecolor='black', facecolor='none',
                                   linestyle='--', label='Physical Array Boundary')
    ax.add_patch(array_rect)

    # 4. Ray trace and plot each angle
    colors = plt.cm.tab10.colors
    for idx, (az, el) in enumerate(test_angles):

        x_hit, y_hit, _, valid_mask = trace_rays_vmap(x_aperture, y_aperture,
                                                      jnp.full_like(x_aperture, az),
                                                      jnp.full_like(y_aperture, el),
                                                      params,
                                                      F, 500)

        hits_x = (x_hit >= -0.5) & (x_hit <= 0.5)
        hits_y = (y_hit >= -2.5) & (y_hit <= 2.5)

        captured_mask = hits_x & hits_y & (valid_mask > 0.5)

        num_total = len(x_hit)
        num_captured = int(jnp.sum(captured_mask))
        efficiency = (num_captured / num_total) * 100

        print(f"Beam Az {az:5.1f}°, El {el:5.1f}° | Rays Captured: {num_captured}/{num_total} | Efficiency: {efficiency:6.2f}%")

        # FIX: Filter the arrays before plotting so shadowed rays are visually dropped
        valid_idx = valid_mask > 0.5
        x_plot = x_hit[valid_idx]
        y_plot = y_hit[valid_idx]

        # Plot the scattered, valid rays
        ax.scatter(x_plot, y_plot, s=2, alpha=0.5, color=colors[idx % len(colors)],
                   label=f'Beam: Az {az}°, El {el}°')

        # Plot the theoretical centroid target we optimized for
        x_target = -(az / 4.0) * 0.35
        y_target = -(el / 15.0) * 1.8
        ax.plot(x_target, y_target, marker='x', color='black', markersize=8, markeredgewidth=2)

    # 5. Formatting (Ensuring 1:1 physical aspect ratio)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.set_xlabel("Array X Dimension (m) - Azimuth Steering")
    ax.set_ylabel("Array Y Dimension (m) - Elevation Steering")
    ax.set_title("GO Ray Footprints on the Tilted Array Face")
    ax.grid(True, alpha=0.3)


    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

    plt.tight_layout()
    plt.show()

def print_capture_efficiencies(params, x_aperture, y_aperture, F):
    test_angles = [
        (0.0, 0.0), (4.0, 15.0), (-4.0, 15.0), (4.0, -15.0),
        (-4.0, -15.0), (4.0, 0.0), (0.0, 15.0)
    ]

    print(f"\n--- Post-Optimization Efficiency (f/D = {F/D:.3f}) ---")

    for az, el in test_angles:
        x_hit, y_hit, _, valid_mask = trace_rays_vmap(
            x_aperture, y_aperture,
            jnp.full_like(x_aperture, az), jnp.full_like(y_aperture, el),
            params, F, 500
        )

        hits_x = (x_hit >= -0.5) & (x_hit <= 0.5)
        hits_y = (y_hit >= -2.5) & (y_hit <= 2.5)

        captured_mask = hits_x & hits_y & (valid_mask > 0.5)

        num_total = len(x_hit)
        num_valid = int(jnp.sum(valid_mask)) 
        num_captured = int(jnp.sum(captured_mask))

        efficiency = (num_captured / num_total) * 100.0
        print(f"  Az {az:5.1f}°, El {el:6.1f}° | Unshadowed: {num_valid:4d}/{num_total:4d} | Captured: {num_captured:4d} | Eff: {efficiency:5.1f}%")



if __name__ == "__main__":
    params_list = []
    focuses = []

    num_spatial = 60
    x_lin = jnp.linspace(X_OFFSET - D/2, X_OFFSET + D/2, num_spatial)
    y_lin = jnp.linspace(Y_OFFSET - D/2, Y_OFFSET + D/2, num_spatial) # <-- Update y_lin
    X, Y = jnp.meshgrid(x_lin, y_lin)

    # <-- Update the mask to use Y_OFFSET
    aperture_mask = ((X - X_OFFSET)**2 + (Y - Y_OFFSET)**2) <= (D/2)**2
    valid_rays_mask = aperture_mask

    x_aperture = X[valid_rays_mask].flatten()
    y_aperture = Y[valid_rays_mask].flatten()


    for f_D in jnp.linspace(0.33, 0.45, 13):
        F = f_D * D
        print(f"\n--- Sweeping f/D = {f_D:.3f} (F = {F:.2f}m) ---")
        params = optimize_reflector(F, x_aperture, y_aperture, X, Y, aperture_mask) # Pass F into the function
        params_list.append(params)
        focuses.append(f_D)
        print_capture_efficiencies(params, x_aperture, y_aperture, F)


    plot_ray_footprints(params_list[4], 0.37 * D) # for f/D = 0.4
