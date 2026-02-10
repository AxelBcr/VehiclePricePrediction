def plot_pca_biplot_adaptive(X_num_scaled, numeric_cols, n_components='auto',
                             sample_size=5000, figsize=(12, 9), interactive=True):
    """
    Adaptive PCA biplot - automatically plots 2D or 3D (interactive) based on n_components

    Parameters:
    -----------
    X_num_scaled : array-like, shape (n_samples, n_features)
        Scaled numerical data
    numeric_cols : list
        Names of the features
    n_components : int or 'auto'
        Number of PCA components. If 'auto', uses min(n_features, 3)
    sample_size : int
        Number of observations to plot (for performance)
    figsize : tuple
        Figure size
    interactive : bool
        If True and n_components >= 3, use plotly for interactivity
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # Determine number of components
    if n_components == 'auto':
        n_components = min(X_num_scaled.shape[1], 3)

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_num_scaled)

    # Sample observations for plotting
    if X_pca.shape[0] > sample_size:
        sample_idx = np.random.choice(X_pca.shape[0], size=sample_size, replace=False)
        X_pca_sample = X_pca[sample_idx]
    else:
        X_pca_sample = X_pca

    # Compute loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 3

    #  2D PLOT (Static matplotlib)
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter observations
        ax.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1],
                   alpha=0.4, s=20, label='Individus', c='lightblue')

        # Loading vectors
        for i, col in enumerate(numeric_cols):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                     head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.7)
            ax.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, col,
                    fontsize=10, ha='center', va='center', color='darkred', weight='bold')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        ax.set_title('Biplot ACP 2D - PC1 vs PC2')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()

    #  3D INTERACTIVE PLOT (Plotly)
    elif n_components == 3 and interactive:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()

            # Scatter observations
            fig.add_trace(go.Scatter3d(
                x=X_pca_sample[:, 0],
                y=X_pca_sample[:, 1],
                z=X_pca_sample[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightblue', opacity=0.4),
                name='Individus',
                hoverinfo='skip'
            ))

            # Loading vectors
            for i, col in enumerate(numeric_cols):
                fig.add_trace(go.Scatter3d(
                    x=[0, loadings[i, 0]],
                    y=[0, loadings[i, 1]],
                    z=[0, loadings[i, 2]],
                    mode='lines+text',
                    line=dict(color='red', width=6),
                    text=['', col],
                    textposition='top center',
                    textfont=dict(size=11, color='darkred'),
                    name=col,
                    hoverinfo='skip',
                    showlegend=True
                ))

            fig.update_layout(
                title='Biplot ACP 3D - PC1 vs PC2 vs PC3 (Interactive)',
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)',
                    xaxis=dict(showgrid=True, zeroline=True),
                    yaxis=dict(showgrid=True, zeroline=True),
                    zaxis=dict(showgrid=True, zeroline=True)
                ),
                width=1000,
                height=800,
                showlegend=True
            )

            fig.show()

        except ImportError:
            print("Plotly not available, falling back to matplotlib 3D")
            interactive = False

    #  3D MATPLOTLIB (Fallback or if interactive=False)
    if n_components == 3 and not interactive:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Scatter observations
        ax.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], X_pca_sample[:, 2],
                   alpha=0.3, s=10, label='Individus', c='lightblue')

        # Loading vectors
        for i, col in enumerate(numeric_cols):
            ax.quiver(0, 0, 0, loadings[i, 0], loadings[i, 1], loadings[i, 2],
                      arrow_length_ratio=0.1, color='red', alpha=0.7, linewidth=2)
            ax.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, loadings[i, 2] * 1.2,
                    col, fontsize=10, color='darkred', weight='bold')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)')
        ax.set_title('Biplot ACP 3D - PC1 vs PC2 vs PC3 (Matplotlib)')
        ax.view_init(elev=20, azim=45)
        ax.legend()
        plt.tight_layout()
        plt.show()

    #  4+ DIMENSIONS: Multiple interactive 3D views
    elif n_components > 3 and interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create subplot titles
            subplot_titles = []
            combinations = []
            for i in range(min(n_components, 3)):
                for j in range(i + 1, min(n_components, 4)):
                    for k in range(j + 1, min(n_components, 5)):
                        subplot_titles.append(f'PC{i + 1}-PC{j + 1}-PC{k + 1}')
                        combinations.append((i, j, k))
                        if len(combinations) >= 4:  # Max 4 subplots
                            break
                    if len(combinations) >= 4:
                        break
                if len(combinations) >= 4:
                    break

            n_plots = len(combinations)
            n_cols = 2
            n_rows = int(np.ceil(n_plots / n_cols))

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=subplot_titles,
                specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )

            for idx, (i, j, k) in enumerate(combinations):
                row = idx // n_cols + 1
                col = idx % n_cols + 1

                # Observations
                fig.add_trace(
                    go.Scatter3d(
                        x=X_pca_sample[:, i],
                        y=X_pca_sample[:, j],
                        z=X_pca_sample[:, k],
                        mode='markers',
                        marker=dict(size=2, color='lightblue', opacity=0.3),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )

                # Loading vectors
                for var_idx, var_col in enumerate(numeric_cols):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[0, loadings[var_idx, i]],
                            y=[0, loadings[var_idx, j]],
                            z=[0, loadings[var_idx, k]],
                            mode='lines',
                            line=dict(color='red', width=4),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )

            fig.update_layout(
                title=f'Biplot ACP {n_components}D - Multiple 3D Views',
                height=400 * n_rows,
                width=1200
            )

            fig.show()

        except ImportError:
            print("Plotly not available, falling back to 2D projections")
            interactive = False

    #  4+ DIMENSIONS: Static 2D projections
    if n_components > 3 and not interactive:
        n_plots = min(6, n_components * (n_components - 1) // 2)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * 1.5, figsize[1]))
        axes = axes.flatten() if n_plots > 1 else [axes]

        plot_idx = 0
        for i in range(min(n_components, 3)):
            for j in range(i + 1, min(n_components, 4)):
                if plot_idx >= n_plots:
                    break

                ax = axes[plot_idx]

                # Scatter
                ax.scatter(X_pca_sample[:, i], X_pca_sample[:, j],
                           alpha=0.4, s=10, c='lightblue')

                # Loadings
                for k, col in enumerate(numeric_cols):
                    ax.arrow(0, 0, loadings[k, i], loadings[k, j],
                             head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
                    ax.text(loadings[k, i] * 1.2, loadings[k, j] * 1.2, col,
                            fontsize=8, ha='center', color='darkred')

                ax.set_xlabel(f'PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)')
                ax.set_ylabel(f'PC{j + 1} ({pca.explained_variance_ratio_[j] * 100:.1f}%)')
                ax.set_title(f'PC{i + 1} vs PC{j + 1}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f'Biplot ACP {n_components}D - Multiple Views', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.show()

    return pca, X_pca