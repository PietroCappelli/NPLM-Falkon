import torch 
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.stats as stats
import matplotlib as mpl

# from analysis_utils import process_nuisance_data, compute_log_ratio, compute_learned_log_ratio_nu


def draw_cms_label(ax: plt.Axes, label: str = "Preliminary", rlabel: str = "NPLM", fontsize: int = 28):
    """
    Draw a CMS label on the given matplotlib Axes object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object to draw the label on.
    label : str, optional
        The label text to display (default is "Preliminary").
    rlabel : str, optional
        The right label text to display (default is "NPLM").
    fontsize : int, optional
        The font size to use for the label (default is 28).
    """
    hep.cms.label(
        ax       = ax,
        data     = True,
        label    = label,
        rlabel   = rlabel,
        fontsize = fontsize
    )
    

def set_label_font(ax: plt.Axes, fontsize: int = 28):
    """
    Set the font size of the x and y axis labels of a matplotlib Axes object.
    
    Parameters:
    ax (matplotlib.pyplot.Axes): The Axes object to modify.
    fontsize (int): The font size to set the labels to. Default is 28.
    """
    ax.set_xlabel(ax.get_xlabel(), fontsize = fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize = fontsize)
    
    
def set_tick_font(ax: plt.Axes, fontsize: int = 28):
    """
    Set the font size of the tick labels for the given Axes object.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object for which to set the tick font size.
    fontsize : int, optional
        The font size to use for the tick labels. Default is 28.
    """
    ax.tick_params(axis = "x", labelsize = fontsize, which = "major")
    ax.tick_params(axis = "y", labelsize = fontsize, which = "major")
    
    
def draw_grid(ax: plt.Axes):
    """
    Draw a grid on the given matplotlib Axes object.
    
    Parameters:
    -----------
    ax: plt.Axes
        The matplotlib Axes object on which to draw the grid.
    """
    ax.grid(True, which="major", axis="both", alpha=0.5, color="gray")
    ax.set_axisbelow(True)


def plot_ref_data(
    feature_ref   : torch.Tensor,
    feature_data  : torch.Tensor,
    weights_ref   : torch.Tensor,
    weights_data  : torch.Tensor,
    figsize       : tuple, 
    bins          : np.ndarray, 
    ratio         : bool = True, 
    h_ratio       : list = [3, 1], 
    fontsize      : int  = 36,
    cms           : bool = False,
    cms_label     : str = "",
    cms_rlabel    : str = "",
    color_ref     : str = "lightgray",
    color_data    : str = "black",
    ref_hist_type : str = "stepfilled",
    ref_label     : str = "Reference",
    data_label    : str = "Data",
    ref_alpha     : float = 1.0,
    xlabel        : str = "Feature",
    ylabel        : str = "Events",
    ratio_ylabel  : str = "Ratio",
    hist_yscale   : str = "linear",
    ratio_yscale  : str = "linear",
    show_plot     : bool = True,
    save_plot     : bool = False,
    plot_name     : str = "ref_data",
    plot_path     : str = "./",
    plot_format   : str = "pdf",
    return_fig    : bool = False,
    constrained   : bool = True,
    ):
    """
    Plots reference and data histograms with ratio plot.
    
    Parameters:
    -----------
    feature_ref : torch.Tensor
        Reference feature tensor.
    feature_data : torch.Tensor
        Data feature tensor.
    weights_ref : torch.Tensor
        Reference weights tensor.
    weights_data : torch.Tensor
        Data weights tensor.
    figsize : tuple
        Figure size.
    bins : np.ndarray
        Bins for the histograms.
    ratio : bool, optional
        Whether to plot the ratio plot, by default True.
    h_ratio : list, optional
        Height ratio of the subplots, by default [3, 1].
    fontsize : int, optional
        Font size, by default 36.
    cms : bool, optional
        Whether to draw CMS label, by default False.
    cms_label : str, optional
        CMS label, by default "".
    cms_rlabel : str, optional
        CMS right label, by default "".
    color_ref : str, optional
        Reference histogram color, by default "lightgray".
    color_data : str, optional
        Data histogram color, by default "black".
    ref_hist_type : str, optional
        Reference histogram type, by default "stepfilled".
    ref_label : str, optional
        Reference histogram label, by default "Reference".
    data_label : str, optional
        Data histogram label, by default "Data".
    ref_alpha : float, optional
        Reference histogram alpha, by default 1.0.
    xlabel : str, optional
        X-axis label, by default "Feature".
    ylabel : str, optional
        Y-axis label, by default "Events".
    ratio_ylabel : str, optional
        Ratio plot Y-axis label, by default "Ratio".
    hist_yscale : str, optional
        Histogram Y-axis scale, by default "linear".
    ratio_yscale : str, optional
        Ratio plot Y-axis scale, by default "linear".
    show_plot : bool, optional
        Whether to show the plot, by default True.
    save_plot : bool, optional
        Whether to save the plot, by default False.
    plot_name : str, optional
        Plot name, by default "ref_data".
    plot_path : str, optional
        Plot path, by default "./".
    plot_format : str, optional
        Plot format, by default "pdf".
    return_fig : bool, optional
        Whether to return the figure and axes objects, by default False.
    """
    if ratio:
        fig, axes = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = constrained, 
            nrows               = 2, 
            ncols               = 1, 
            sharex              = True, 
            gridspec_kw         = {"height_ratios": h_ratio}
        )
        ax_hist  = axes[0]
        ax_ratio = axes[1]
    else:
        fig, ax_hist = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = constrained
        )
        
    ax_hist.set_xlim(bins[0], bins[-1])
    
    bincenters = 0.5*(bins[1:]+bins[:-1])
    
    if cms:
        draw_cms_label(ax=ax_hist, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax_hist, fontsize=fontsize)
    set_tick_font(ax=ax_hist, fontsize=fontsize-4)

    
    # histograms
    ref_hist,  _ = np.histogram(feature_ref.flatten(),  bins=bins, weights=weights_ref.flatten())
    data_hist, _ = np.histogram(feature_data.flatten(), bins=bins, weights=weights_data.flatten())
    
    # reference
    ax_hist.hist(bincenters, bins=bins, weights=ref_hist,  histtype=ref_hist_type, label=ref_label, color=color_ref, alpha=ref_alpha)

    # data
    ax_hist.errorbar(
        x          = bincenters, 
        y          = data_hist, 
        yerr       = np.sqrt(data_hist), 
        marker     = "o",
        ls         = "",
        lw         = 2,
        label      = data_label,
        color      = color_data,
        markersize = 10,
        elinewidth = 2,
        capsize    = 5,
        capthick   = 2
    )
    
    ax_hist.legend(fontsize=fontsize-4)
    ax_hist.set_ylabel(ylabel, fontsize=fontsize)
    ax_hist.set_yscale(hist_yscale)
    if not ratio:
        ax_hist.set_xlabel(xlabel, fontsize=fontsize)
    
    if ratio:
        y_ratio = data_hist / ref_hist
        y_err_ratio = np.sqrt(data_hist) / ref_hist
        y_mask = y_ratio != 0
        y_ratio     = y_ratio[y_mask]
        y_err_ratio = y_err_ratio[y_mask]
        bincenters_ratio  = bincenters[y_mask]
    
        ax_ratio.axhline(y=1, ls="--", color="black", lw=2)
        set_label_font(ax=ax_ratio, fontsize=fontsize)
        set_tick_font(ax=ax_ratio, fontsize=fontsize-4)
        
        ax_ratio.errorbar(
            x          = bincenters_ratio, 
            y          = y_ratio,
            yerr       = y_err_ratio,
            marker     = "o",
            ls         = "",
            lw         = 2,
            color      = color_data, 
            markersize = 10,
            elinewidth = 2,
            capsize    = 5,
            capthick   = 2
        )

        ax_ratio.set_xlabel(xlabel, fontsize=fontsize)
        ax_ratio.set_ylabel(ratio_ylabel, fontsize=fontsize)
        
        if ratio_yscale == "log":
            ax_ratio.set_yscale(ratio_yscale)
        elif ratio_yscale == "linear":
            ratio_yabs_max = np.max(np.abs(ax_ratio.get_ylim())) 
            ax_ratio.set_ylim(0, ratio_yabs_max)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, axes
    
    return


def plot_ref_data_reco(
    feature_ref    : torch.Tensor,
    feature_data   : torch.Tensor,
    weights_ref    : torch.Tensor,
    weights_data   : torch.Tensor,
    prediction_ref : torch.Tensor,
    figsize        : tuple, 
    bins           : np.ndarray, 
    ratio          : bool         = True, 
    h_ratio        : list         = [3, 1], 
    fontsize       : int          = 36,
    cms            : bool         = False,
    cms_label      : str          = "",
    cms_rlabel     : str          = "",
    color_ref      : str          = "lightgray",
    color_data     : str          = "black",
    ref_hist_type  : str          = "stepfilled",
    pred_hist_type : str          = "step",
    ref_label      : str          = "Reference",
    data_label     : str          = "Data",
    ref_alpha      : float        = 1.0,
    color_reco     : str          = "red",
    reco_label     : str          = "Reco",
    lw_reco        : float        = 3,
    alpha_reco     : float        = 1.0,
    xlabel         : str          = "Feature",
    ylabel         : str          = "Events",
    ratio_ylabel   : str          = "Ratio",
    hist_yscale    : str          = "linear",
    ratio_yscale   : str          = "linear",
    show_plot      : bool         = True,
    save_plot      : bool         = False,
    plot_name      : str          = "ref_data",
    plot_path      : str          = "./",
    plot_format    : str          = "pdf",
    return_fig     : bool         = False,
    binned_reco    : bool         = False,
    grid_reco      : torch.Tensor = None,
    grid_pred      : torch.Tensor = None,
    constrained    : bool         = True,
    ):
    """
    Plots the reference, data and reconstructed histograms for a given feature.
    
    Args:
    - feature_ref: torch.Tensor of shape (n_samples_ref, n_features) containing the reference feature values.
    - feature_data: torch.Tensor of shape (n_samples_data, n_features) containing the data feature values.
    - weights_ref: torch.Tensor of shape (n_samples_ref,) containing the weights for the reference samples.
    - weights_data: torch.Tensor of shape (n_samples_data,) containing the weights for the data samples.
    - prediction_ref: torch.Tensor of shape (n_samples_ref,) containing the reconstructed feature values for the reference samples.
    - figsize: tuple of integers (width, height) in inches specifying the size of the figure.
    - bins: np.ndarray of shape (n_bins+1,) containing the edges of the bins for the histograms.
    - ratio: boolean indicating whether to plot the ratio of data to reference histograms.
    - h_ratio: list of two integers specifying the height ratio between the histogram and ratio plots.
    - fontsize: integer specifying the font size for the labels and ticks.
    - cms: boolean indicating whether to add the CMS label to the plot.
    - cms_label: string specifying the CMS label text.
    - cms_rlabel: string specifying the CMS label text on the right side of the plot.
    - color_ref: string specifying the color for the reference histogram.
    - color_data: string specifying the color for the data histogram.
    - ref_hist_type: string specifying the histogram type for the reference histogram.
    - pred_hist_type: string specifying the histogram type for the reconstructed histogram.
    - ref_label: string specifying the label for the reference histogram.
    - data_label: string specifying the label for the data histogram.
    - ref_alpha: float specifying the alpha value for the reference histogram.
    - color_reco: string specifying the color for the reconstructed histogram.
    - reco_label: string specifying the label for the reconstructed histogram.
    - lw_reco: float specifying the line width for the reconstructed histogram.
    - alpha_reco: float specifying the alpha value for the reconstructed histogram.
    - xlabel: string specifying the label for the x-axis.
    - ylabel: string specifying the label for the y-axis.
    - ratio_ylabel: string specifying the label for the y-axis of the ratio plot.
    - hist_yscale: string specifying the y-axis scale for the histogram plot.
    - ratio_yscale: string specifying the y-axis scale for the ratio plot.
    - show_plot: boolean indicating whether to show the plot.
    - save_plot: boolean indicating whether to save the plot.
    - plot_name: string specifying the name of the plot file.
    - plot_path: string specifying the path to save the plot file.
    - plot_format: string specifying the format of the plot file.
    - return_fig: boolean indicating whether to return the figure and axes objects.
    - binned_reco: boolean indicating whether to plot the reconstructed histogram as binned data.
    - grid_reco: torch.Tensor of shape (n_bins_reco,) containing the bin centers for the reconstructed histogram.
    - grid_pred: torch.Tensor of shape (n_bins_reco,) containing the reconstructed feature values for the bin centers.
    
    Returns:
    - If return_fig is True, returns the figure and axes objects.
    - Otherwise, returns None.
    """
    if ratio:
        fig, axes = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = constrained, 
            nrows               = 2, 
            ncols               = 1, 
            sharex              = True, 
            gridspec_kw         = {"height_ratios": h_ratio}
        )
        ax_hist  = axes[0]
        ax_ratio = axes[1]
    else:
        fig, ax_hist = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = constrained
        )
        
    ax_hist.set_xlim(bins[0], bins[-1])
    
    bincenters = 0.5*(bins[1:]+bins[:-1])
    
    if cms:
        draw_cms_label(ax=ax_hist, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax_hist, fontsize=fontsize)
    set_tick_font(ax=ax_hist, fontsize=fontsize-4)

    
    # histograms
    ref_hist,  _ = np.histogram(feature_ref.flatten(),  bins=bins, weights=weights_ref.flatten())
    data_hist, _ = np.histogram(feature_data.flatten(), bins=bins, weights=weights_data.flatten())
    pred_hist, _ = np.histogram(feature_ref.flatten(), bins=bins, weights=weights_ref.flatten()*np.exp(prediction_ref.flatten()))
    
    # reference
    ax_hist.hist(bincenters, bins=bins, weights=ref_hist,  histtype=ref_hist_type, label=ref_label, color=color_ref, alpha=ref_alpha)
    
    # reco
    ax_hist.hist(bincenters, bins=bins, weights=pred_hist, histtype=pred_hist_type, label=reco_label, color=color_reco, lw=lw_reco, alpha=alpha_reco)

    # data
    ax_hist.errorbar(
        x          = bincenters, 
        y          = data_hist, 
        yerr       = np.sqrt(data_hist), 
        marker     = "o",
        ls         = "",
        lw         = 2,
        label      = data_label,
        color      = color_data,
        markersize = 10,
        elinewidth = 2,
        capsize    = 5,
        capthick   = 2
    )
    
    ax_hist.legend(fontsize=fontsize-4)
    ax_hist.set_ylabel(ylabel, fontsize=fontsize)
    ax_hist.set_yscale(hist_yscale)
    if not ratio:
        ax_hist.set_xlabel(xlabel, fontsize=fontsize)
    
    if ratio:
        y_ratio = data_hist / ref_hist
        y_err_ratio = np.sqrt(data_hist) / ref_hist
        y_mask = y_ratio != 0
        y_ratio     = y_ratio[y_mask]
        y_err_ratio = y_err_ratio[y_mask]
        bincenters_ratio  = bincenters[y_mask]
        
        if binned_reco:
            y_reco = pred_hist / ref_hist

            ax_ratio.plot(
                bincenters,
                y_reco,
                ls  = "-",
                lw = 6,
                color = color_reco,
                alpha = alpha_reco
            )
            
        elif grid_reco is not None and grid_pred is not None and not binned_reco:
            y_reco = np.exp(grid_pred) 
            
            ax_ratio.plot(
                grid_reco,
                y_reco,
                ls  = "-",
                lw = 6,
                color = color_reco,
                alpha = alpha_reco
            )
            

        ax_ratio.axhline(y=1, ls="--", color="black", lw=2)
        set_label_font(ax=ax_ratio, fontsize=fontsize)
        set_tick_font(ax=ax_ratio, fontsize=fontsize-4)
        
        ax_ratio.errorbar(
            x          = bincenters_ratio, 
            y          = y_ratio,
            yerr       = y_err_ratio,
            marker     = "o",
            ls         = "",
            lw         = 2,
            color      = color_data, 
            markersize = 10,
            elinewidth = 2,
            capsize    = 5,
            capthick   = 2
        )

        ax_ratio.set_xlabel(xlabel, fontsize=fontsize)
        ax_ratio.set_ylabel(ratio_ylabel, fontsize=fontsize)
        
        if ratio_yscale == "log":
            ax_ratio.set_yscale(ratio_yscale)
        elif ratio_yscale == "linear":
            ratio_yabs_max = np.max(np.abs(ax_ratio.get_ylim())) 
            ax_ratio.set_ylim(-2, ratio_yabs_max+3)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, axes
    
    return


def plot_loss_history(
    n_epochs     : np.ndarray,
    loss_history : np.ndarray,
    epochs_init  : int = 350,
    epochs_norm  : int = 1,
    figsize      : tuple = (14, 10),
    fontsize     : int = 36,
    color        : str = "black",
    lw           : int = 4,
    cms          : bool = False,
    cms_label    : str = "",
    cms_rlabel   : str = "",
    grid         : bool = True,
    xlabel       : str = "Epoch",
    ylabel       : str = "Loss",
    show_plot    : bool = True,
    save_plot    : bool = False,
    plot_name    : str = "loss_history",
    plot_path    : str = "./",
    plot_format  : str = "pdf",
    return_fig   : bool = False,
    constrained  : bool = True,
    ):
    """
    Plots the loss history of a neural network training process.

    Args:
    - n_epochs (np.ndarray): An array containing the number of epochs.
    - loss_history (np.ndarray): An array containing the loss history.
    - epochs_init (int): The initial epoch to start plotting from. Default is 350.
    - epochs_norm (int): The normalization factor for the x-axis. Default is 1.
    - figsize (tuple): The size of the figure. Default is (14, 10).
    - fontsize (int): The font size of the labels. Default is 36.
    - color (str): The color of the plot. Default is "black".
    - lw (int): The line width of the plot. Default is 4.
    - cms (bool): Whether to draw the CMS label. Default is False.
    - cms_label (str): The CMS label. Default is "".
    - cms_rlabel (str): The CMS right label. Default is "".
    - grid (bool): Whether to draw the grid. Default is True.
    - xlabel (str): The label for the x-axis. Default is "Epoch".
    - ylabel (str): The label for the y-axis. Default is "Loss".
    - show_plot (bool): Whether to show the plot. Default is True.
    - save_plot (bool): Whether to save the plot. Default is False.
    - plot_name (str): The name of the plot. Default is "loss_history".
    - plot_path (str): The path to save the plot. Default is "./".
    - plot_format (str): The format of the plot. Default is "pdf".
    - return_fig (bool): Whether to return the figure and axis objects. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object. Only returned if return_fig is True.
    - ax (matplotlib.axes.Axes): The axis object. Only returned if return_fig is True.
    """
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
    
    if cms:
        draw_cms_label(ax=ax, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    if grid:
        draw_grid(ax=ax)
    set_label_font(ax=ax, fontsize=fontsize)
    set_tick_font(ax=ax, fontsize=fontsize)

    ax.plot(np.arange(epochs_init, n_epochs) / epochs_norm, loss_history[epochs_init:], lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white") 

    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, ax
    
    

def plot_one_t(
    t_distribution : np.ndarray,
    t_bins         : np.ndarray,
    chi2           : stats._distn_infrastructure.rv_continuous_frozen,
    chi2_grid      : np.ndarray,
    show_hist      : bool = True,
    show_error     : bool = False,
    compute_rate   : bool = False,
    err_marker     : str  = "o",
    err_markersize : float = 10,
    err_capsize    : float = 5,
    err_elinewidth : float = 2,
    err_capthick   : float = 2,
    err_color      : str   = "black",
    figsize        : tuple = (14, 10),
    fontsize       : int   = 36,
    cms            : bool  = False,
    cms_label      : str   = "",
    cms_rlabel     : str   = "",
    hist_ecolor    : str   = "black",
    hist_fcolor    : str   = "black",
    chi2_color     : str   = "red",
    hist_lw        : float = 4,
    chi2_lw        : float = 4,
    hist_type      : str   = "step",
    hist_label     : str   = "Data",
    chi2_label     : str   = r"$\chi^2$",
    xlabel         : str   = r"$t$",
    ylabel         : str   = "Density",
    show_plot      : bool  = False,
    save_plot      : bool  = False,
    plot_name      : str   = "t_distribution",
    plot_path      : str   = "./",
    plot_format    : str   = "pdf",
    return_fig     : bool  = False,
    constrained    : bool  = True,
    plot_params    : bool  = False,
    hyperparams    : str   = "",
    ):  

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
    
    if cms:
        draw_cms_label(ax=ax, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax, fontsize=fontsize)
    set_tick_font(ax=ax, fontsize=fontsize-4)

    
    t_bincenters = (t_bins[1:] + t_bins[:-1]) / 2
    t_binwidths   = (t_bins[1:] - t_bins[:-1])
    
    hist_, _  = np.histogram(t_distribution, bins=t_bins, density=False)
    hist_norm = np.sum([hist_[i] * t_binwidths[i] for i in range(t_bincenters.size)]) 
    hist      = hist_ / hist_norm   
    hist_err  = np.sqrt(hist / hist_norm)
    
    if compute_rate:
        n_bins    = t_bincenters.size
        n_obs     = t_distribution.size
        exp_rate  = (n_obs / n_bins) / hist_norm
        exp_err   = np.sqrt(exp_rate / hist_norm)

        hist_err  = [hist_err[i] if hist_err[i] > 0 else exp_err for i in range(n_bins)]
    else:
        mask_hist_not_zero = (hist != 0)
        t_bincenters       = t_bincenters[mask_hist_not_zero]
        hist               = hist[mask_hist_not_zero]
        hist_err           = hist_err[mask_hist_not_zero]
        
    if show_hist:
        if hist_type == "step":
            h = ax.hist(
                t_bincenters,
                bins     = t_bins,
                weights  = hist,
                histtype = hist_type,
                label    = hist_label,
                color    = hist_ecolor,
                lw       = hist_lw,
            )[-1][0]
        
        if hist_type == "stepfilled":
            h = ax.hist(
                t_bincenters,
                bins      = t_bins,
                weights   = hist,
                histtype  = hist_type,
                label     = hist_label,
                edgecolor = hist_ecolor,
                facecolor = hist_fcolor,
                lw        = hist_lw,
            )[-1][0]
    
    if show_error:
        err = ax.errorbar(
            x          = t_bincenters,
            y          = hist,
            yerr       = hist_err,
            marker     = err_marker,
            markersize = err_markersize,
            capsize    = err_capsize,
            elinewidth = err_elinewidth,
            capthick   = err_capthick,
            color      = err_color,
            ls         = "",
        )
        
    chisq = ax.plot(
        chi2_grid,
        chi2.pdf(chi2_grid),
        label = chi2_label,
        color = chi2_color,
        lw    = chi2_lw,
    )[0]
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(t_bins[0], t_bins[-1])
    
    if show_hist:
        if plot_params: 
            ax.legend([chisq, h], [chi2_label, hist_label], fontsize=fontsize-6, title = "$(\lambda, M)$ = ("+ hyperparams+")", title_fontsize = fontsize-4)
        else:
            ax.legend([chisq, h], [chi2_label, hist_label], fontsize=fontsize-6)
    elif not show_hist and show_error:
        if plot_params:
            ax.legend([chisq, err], [chi2_label, hist_label], fontsize=fontsize-6, title = "$(\lambda, M)$ = ("+ hyperparams+")", title_fontsize = fontsize-4)
        else: 
            ax.legend([chisq, err], [chi2_label, hist_label], fontsize=fontsize-6)
    else:
        if plot_params: 
            ax.legend([chisq], [chi2_label], fontsize=fontsize-6, title = "$(\lambda, M)$ = ("+ hyperparams+")")
        else: 
            ax.legend([chisq], [chi2_label], fontsize=fontsize-5)
            
            
    ax.axvline(x=np.median(t_distribution), color='gray', linestyle='--', linewidth=4, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, ax
    
    return


def plot_two_t(
    t_distribution_1   : np.ndarray,
    t_bins_1           : np.ndarray,
    t_distribution_2   : np.ndarray,
    t_bins_2           : np.ndarray,
    chi2               : stats._distn_infrastructure.rv_continuous_frozen,
    chi2_grid          : np.ndarray,
    show_error_1       : bool = False,
    show_error_2       : bool = False,
    show_hist_1        : bool = True,
    show_hist_2        : bool = True,
    compute_rate_1     : bool = False,
    compute_rate_2     : bool = False,
    err_marker_1       : str  = "o",
    err_marker_2       : str  = "o",
    err_markersize_1   : float = 10,
    err_markersize_2   : float = 10,
    err_capsize_1      : float = 5,
    err_capsize_2      : float = 5,
    err_elinewidth_1   : float = 2,
    err_elinewidth_2   : float = 2,
    err_capthick_1     : float = 2,
    err_capthick_2     : float = 2,
    err_ecolor_1       : str   = "black",
    err_ecolor_2       : str   = "black",
    figsize            : tuple = (14, 10),
    fontsize           : int   = 36,
    cms                : bool  = False,
    cms_label          : str   = "",
    cms_rlabel         : str   = "",
    hist_ecolor_1      : str   = "black",
    hist_ecolor_2      : str   = "black",
    hist_fcolor_1      : str   = "black",
    hist_fcolor_2      : str   = "black",
    chi2_color         : str   = "red",
    hist_lw_1          : float = 4,
    hist_lw_2          : float = 4,
    chi2_lw            : float = 4,
    hist_type_1        : str   = "step",
    hist_type_2        : str   = "step",
    hist_label_1       : str   = "Data",
    hist_label_2       : str   = "Data",
    chi2_label         : str   = r"$\chi^2$",
    xlabel             : str   = r"$t$",
    ylabel             : str   = "Density",
    show_plot          : bool  = True,
    save_plot          : bool  = False,
    plot_name          : str   = "t_distribution",
    plot_path          : str   = "./",
    plot_format        : str   = "pdf",
    return_fig         : bool  = False,
    constrained        : bool  = True,
    plot_params        : bool  = False,
    hyperparams        : str   = "",
    N_SIG              : int   = None,
    ):  

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
    
    if cms:
        draw_cms_label(ax=ax, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax, fontsize=fontsize)
    set_tick_font(ax=ax, fontsize=fontsize-4)

    
    t_bincenters_1  = (t_bins_1[1:] + t_bins_1[:-1]) / 2
    t_binwidths_1   = (t_bins_1[1:] - t_bins_1[:-1])
    
    t_bincenters_2  = (t_bins_2[1:] + t_bins_2[:-1]) / 2
    t_binwidths_2   = (t_bins_2[1:] - t_bins_2[:-1])
    
    hist_1_, _  = np.histogram(t_distribution_1, bins=t_bins_1, density=False)
    hist_norm_1 = np.sum([hist_1_[i] * t_binwidths_1[i] for i in range(t_bincenters_1.size)]) 
    hist_1      = hist_1_ / hist_norm_1   
    hist_err_1  = np.sqrt(hist_1 / hist_norm_1)
    
    hist_2_, _  = np.histogram(t_distribution_2, bins=t_bins_2, density=False)
    hist_norm_2 = np.sum([hist_2_[i] * t_binwidths_2[i] for i in range(t_bincenters_2.size)]) 
    hist_2      = hist_2_ / hist_norm_2   
    hist_err_2  = np.sqrt(hist_2 / hist_norm_2)
    
    if compute_rate_1:
        n_bins_1    = t_bincenters_1.size
        n_obs_1     = t_distribution_1.size
        exp_rate_1  = (n_obs_1 / n_bins_1) / hist_norm_1
        exp_err_1   = np.sqrt(exp_rate_1 / hist_norm_1)
        
        hist_err_1  = [hist_err_1[i] if hist_err_1[i] > 0 else exp_err_1 for i in range(n_bins_1)]
        
    elif not compute_rate_1:
        mask_hist_not_zero_1 = (hist_1 != 0)
        t_bincenters_1       = t_bincenters_1[mask_hist_not_zero_1]
        hist_1               = hist_1[mask_hist_not_zero_1]
        hist_err_1           = hist_err_1[mask_hist_not_zero_1]
        
    if compute_rate_2:
        n_bins_2    = t_bincenters_2.size
        n_obs_2     = t_distribution_2.size
        exp_rate_2  = (n_obs_2 / n_bins_2) / hist_norm_2
        exp_err_2   = np.sqrt(exp_rate_2 / hist_norm_2)

        hist_err_2  = [hist_err_2[i] if hist_err_2[i] > 0 else exp_err_2 for i in range(n_bins_2)]
        
    elif not compute_rate_2:
        mask_hist_not_zero_2 = (hist_2 != 0)
        t_bincenters_2       = t_bincenters_2[mask_hist_not_zero_2]
        hist_2               = hist_2[mask_hist_not_zero_2]
        hist_err_2           = hist_err_2[mask_hist_not_zero_2]
        
    
    if show_hist_1:
        if hist_type_1 == "step":
            h_1 = ax.hist(
                t_bincenters_1,
                bins     = t_bins_1,
                weights  = hist_1,
                histtype = hist_type_1,
                label    = hist_label_1,
                color    = hist_ecolor_1,
                lw       = hist_lw_1,
            )[-1][0]
        
        if hist_type_1 == "stepfilled":
            h_1 = ax.hist(
                t_bincenters_1,
                bins      = t_bins_1,
                weights   = hist_1,
                histtype  = hist_type_1,
                label     = hist_label_1,
                edgecolor = hist_ecolor_1,
                facecolor = hist_fcolor_1,
                lw        = hist_lw_1,
            )[-1][0]
    
    if show_hist_2:   
        if hist_type_2 == "step":
            h_2 = ax.hist(
                t_bincenters_2,
                bins     = t_bins_2,
                weights  = hist_2,
                histtype = hist_type_2,
                label    = hist_label_2,
                color    = hist_ecolor_2,
                lw       = hist_lw_2,
            )[-1][0]
        
        if hist_type_2 == "stepfilled":
            h_2 = ax.hist(
                t_bincenters_2,
                bins      = t_bins_2,
                weights   = hist_2,
                histtype  = hist_type_2,
                label     = hist_label_2,
                edgecolor = hist_ecolor_2,
                facecolor = hist_fcolor_2,
                lw        = hist_lw_2,
            )[-1][0]
    
    if show_error_1:
        err_1 = ax.errorbar(
            x          = t_bincenters_1,
            y          = hist_1,
            yerr       = hist_err_1,
            marker     = err_marker_1,
            markersize = err_markersize_1,
            capsize    = err_capsize_1,
            elinewidth = err_elinewidth_1,
            capthick   = err_capthick_1,
            color      = err_ecolor_1,
            ls         = "",
            label      = hist_label_1,
        )
    
    if show_error_2:
        err_2 = ax.errorbar(
        x          = t_bincenters_2,
        y          = hist_2,
        yerr       = hist_err_2,
        marker     = err_marker_2,
        markersize = err_markersize_2,
        capsize    = err_capsize_2,
        elinewidth = err_elinewidth_2,
        capthick   = err_capthick_2,
        color      = err_ecolor_2,
        ls         = "",
        label      = hist_label_2,
    )
        
    chisq = ax.plot(
        chi2_grid,
        chi2.pdf(chi2_grid),
        label = chi2_label,
        color = chi2_color,
        lw    = chi2_lw,
        zorder=10
    )[0]
    
    t_obs = ax.axvline(x=np.median(t_distribution_2), color="#D8707C", linestyle='--', linewidth=4, alpha=1, label="t obs")
    
    if show_hist_1 and show_hist_2:
        if plot_params:
            ax.legend([chisq, h_1, h_2, t_obs], [chi2_label, hist_label_1, hist_label_2, "t_obs"], fontsize=fontsize-6, title = "N(S) = "+str(N_SIG) , title_fontsize = fontsize-4)
            # ax.legend([chisq, h_1, h_2], [chi2_label, hist_label_1, hist_label_2], fontsize=fontsize-6, title = "$(\lambda, M)$ = ("+ hyperparams+") N(S) = "+str(N_SIG) , title_fontsize = fontsize-4)
        else:
            ax.legend([chisq, h_1, h_2], [chi2_label, hist_label_1, hist_label_2], fontsize=fontsize-6)
    if show_hist_1 and not show_hist_2 and show_error_2:
        ax.legend([chisq, h_1, err_2], [chi2_label, hist_label_1, hist_label_2], fontsize=fontsize-6)
    if not show_hist_1 and show_hist_2 and show_error_1:
        ax.legend([chisq, err_1, h_2], [chi2_label, hist_label_1, hist_label_2], fontsize=fontsize-6)
    if not show_hist_1 and show_hist_2 and not show_error_1:
        ax.legend([chisq, h_2], [hist_label_2], fontsize=fontsize-6)
    if show_hist_1 and not show_hist_2 and not show_error_2:
        ax.legend([chisq, h_1], [chi2_label, hist_label_1], fontsize=fontsize-6)
    if not show_hist_1 and not show_hist_2 and show_error_1 and show_error_2:
        ax.legend([chisq, err_1, err_2], [chi2_label, hist_label_1, hist_label_2], fontsize=fontsize-6)
    if not show_hist_1 and not show_hist_2 and show_error_1 and not show_error_2:
        ax.legend([chisq, err_1], [chi2_label, hist_label_1], fontsize=fontsize-6)
    if not show_hist_1 and not show_hist_2 and not show_error_1 and show_error_2:
        ax.legend([chisq, err_2], [chi2_label, hist_label_2], fontsize=fontsize-6)

    
    ax.set_ylim(bottom=0)
    
    xmin = min(t_bins_1[0],  t_bins_2[0])
    xmax = max(t_bins_1[-1], t_bins_2[-1])

    ax.set_xlim(xmin, xmax)
        
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, ax
    
    return


# def plot_quantiles_evolution(
#     t_history,
#     quantile_list,
#     quantile_labels,
#     chi2,
#     obs_alpha    : float = 1.0,
#     th_alpha     : float = 1.0,
#     epochs_init  : int = 1000,
#     epochs_norm  : int = 1,
#     figsize      : tuple = (14, 10),
#     fontsize     : int = 36,
#     palette      : list = ["#1f77b4", "#8f6fc6", "#e657a3", "#ff5d58", "#ff7f0e"],
#     lw           : int = 4,
#     cms          : bool = False,
#     cms_label    : str = "",
#     cms_rlabel   : str = "",
#     grid         : bool = True,
#     xlabel       : str = "Epoch",
#     ylabel       : str = "Loss",
#     show_plot    : bool = True,
#     save_plot    : bool = False,
#     plot_name    : str = "loss_history",
#     plot_path    : str = "./",
#     plot_format  : str = "pdf",
#     return_fig   : bool = False,
#     constrained  : bool = True,
# ):
#     fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
    
#     if cms:
#         draw_cms_label(ax=ax, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
#     if grid:
#         draw_grid(ax=ax)
#     set_label_font(ax=ax, fontsize=fontsize)
#     set_tick_font(ax=ax, fontsize=fontsize-4)

    
#     theoretical_quantiles = chi2.ppf(quantile_list)
#     observed_quantiles    = np.quantile(t_history, quantile_list, axis=0)
    
    
#     xmin = epochs_init        / epochs_norm
#     xmax = t_history.shape[1] / epochs_norm
    
#     ymin = 0
    
#     obs_max = np.max(observed_quantiles)
#     th_max  = np.max(theoretical_quantiles)
    
#     if obs_max > th_max:
#         ymax = obs_max + (obs_max - ymin) * 0.2
#     else:
#         ymax = th_max + (th_max - ymin) * 0.2
    
    
#     for i in range(len(quantile_list)):
    
#         # Plot the theoretical quantiles
#         ax.hlines(
#             y         = theoretical_quantiles[i],
#             xmin      = xmin,
#             xmax      = xmax,
#             color     = palette[i],
#             linestyle = "--",
#             linewidth = lw,
#             alpha     = th_alpha,
#             zorder    = 0
#         )
        
#         # Plot the observed quantiles history
#         ax.plot(
#             np.arange(epochs_init, observed_quantiles.shape[1]) / epochs_norm,
#             observed_quantiles[i][epochs_init:],
#             color     = palette[i],
#             linestyle = "-",
#             linewidth = lw,
#             alpha     = obs_alpha,
#             zorder    = 1
#         )
        
#         ax.text(
#             x                   = xmax + 0.005 * (xmax - xmin), 
#             y                   = theoretical_quantiles[i],
#             s                   = quantile_labels[i], 
#             horizontalalignment = "left", 
#             verticalalignment   = "center", 
#             fontsize            = fontsize-4,
#             color               = palette[i],
#             transform           = ax.transData
#         )
    
    
#     ax.set_ylim(ymin, ymax)
#     ax.set_xlim(xmin, xmax)

#     # add fake legend with dashed line labeled "Expected" and solid line labeled "Observed"
#     dashed_line = mpl.lines.Line2D([], [], color=palette[0], linestyle="--", linewidth=4)
#     solid_line  = mpl.lines.Line2D([], [], color=palette[0], linestyle="-",  linewidth=4)

#     ax.legend(
#         [dashed_line, solid_line],
#         ["Expected quantiles", "Observed quantiles"],
#         fontsize = fontsize-6,
#         ncol     = 2,
#         loc      = "upper center"
#     )
    
#     ax.set_xlabel(xlabel, fontsize=fontsize)
#     ax.set_ylabel(ylabel, fontsize=fontsize)

#     if save_plot:
#         fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white") 

#     if show_plot:
#         plt.show()
        
#     if return_fig:
#         return fig, ax
    
    
# def plot_nuisance_variations(
#     feature, 
#     targets, 
#     weights, 
#     nuisance, 
#     nu_list, 
#     nu_list_std, 
#     bins, 
#     figsize, 
#     fontsize, 
#     palette,
#     grid=True,
#     central_value=True,
#     color_centr_val="black",
#     xlabel="x",
#     ylabel="Events",
#     xscale="linear",
#     yscale="log",
#     lw=2,
#     xlims=None,
#     ylims=None,
#     return_fig=False,
#     constrained=True,
#     ):
#     """
#     Plots the variations in data due to different nuisance parameters.

#     Args:
#         feature (np.ndarray): The feature array.
#         targets (np.ndarray): The targets array.
#         weights (np.ndarray): The weights array.
#         nuisance (np.ndarray): The nuisance parameters array.
#         nu_list (np.ndarray): Array of nuisance parameter values.
#         nu_list_std (np.ndarray): Standardized nuisance parameter values.
#         bins (np.ndarray): Bin edges for histogram.
#         figsize (tuple): Size of the figure.
#         fontsize (int): Font size for labels.
#         palette (list): List of colors for different nuisance parameters.
#         central_value (bool, optional): Whether to plot the central value reference hypothesis. Defaults to True.
#         color_centr_val (str, optional): Color of the central value reference hypothesis. Defaults to "black".
#         xlabel (str, optional): Label for the x-axis. Defaults to "x".
#         ylabel (str, optional): Label for the y-axis. Defaults to "Events".
#         xscale (str, optional): Scale of the x-axis. Defaults to "linear".
#         yscale (str, optional): Scale of the y-axis. Defaults to "log".
#         lw (int, optional): Line width of the histograms. Defaults to 2.
#     """
    
#     # create figure and axes
#     fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
#     if grid:
#         draw_grid(ax)
#     set_label_font(ax, fontsize)
#     set_tick_font(ax, fontsize - 2)

#     if central_value:
#         # extraact central value reference hypothesis and rescale to nuisance variations
#         x_ref = feature[(targets == 0)][:, 0]
#         w_ref = weights[(targets == 0)] * 1. / len(nu_list)

#         # plot the central value reference hypothesis
#         ax.hist(x_ref, bins=bins, weights=w_ref, histtype="step", ls="--", lw=lw, label="central value", color=color_centr_val)

#     for nu_iter in range(len(nu_list)):
#         nu     = nu_list[nu_iter]
#         nu_std = nu_list_std[nu_iter]

#         # extract the data for the current nuisance parameter value
#         x_nu = feature[(nuisance == nu_std) & (targets == 1)][:, 0]
#         w_nu = weights[(nuisance == nu_std) & (targets == 1)]

#         ax.hist(x_nu, bins=bins, weights=w_nu, histtype="step", lw=lw, label=f"$\\nu$ = {nu}$\sigma$", color=palette[nu_iter])
    
#     ax.legend(fontsize=fontsize - 4, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    
#     ax.set_xscale(xscale)
#     ax.set_yscale(yscale)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if isinstance(xlims, tuple):
#         ax.set_xlim(xlims)
#     if isinstance(ylims, tuple):
#         ax.set_ylim(ylims)
        
#     plt.show()
        
#     if return_fig:
#         return fig, ax
    

# def plot_nuisance_ratio(
#     feature, 
#     targets, 
#     weights, 
#     nuisance, 
#     nu_list, 
#     nu_list_std, 
#     bins, 
#     figsize, 
#     fontsize, 
#     palette,
#     grid=True,
#     zero_line=True,
#     marker="o",
#     markersize=10,
#     elinewidth=2,
#     capsize=5,
#     capthick=2,
#     xlabel="x",
#     ylabel=r'$\log[n(x|\nu) / n(x|0)]$',
#     xscale="linear",
#     yscale="linear",
#     xlims=None,
#     ylims=None,
#     analytic=None,
#     return_fig=False,
#     constrained=True,
#     ):
#     """
#     Plots the ratio of nuisance variations compared to the reference.

#     Args:
#         feature (np.ndarray): The feature array.
#         targets (np.ndarray): The targets array.
#         weights (np.ndarray): The weights array.
#         nuisance (np.ndarray): The nuisance parameters array.
#         nu_list (np.ndarray): Array of nuisance parameter values.
#         nu_list_std (np.ndarray): Standardized nuisance parameter values.
#         bins (np.ndarray): Bin edges for histogram.
#         figsize (tuple): Size of the figure.
#         fontsize (int): Font size for labels.
#         palette (list): List of colors for different nuisance parameters.
#         grid (bool, optional): Whether to draw the grid. Defaults to True.
#         zero_line (bool, optional): Whether to draw the zero line. Defaults to True.
#         marker (str, optional): Marker for the error bars. Defaults to "o".
#         markersize (int, optional): Marker size for the error bars. Defaults to 10.
#         elinewidth (int, optional): Line width for the error bars. Defaults to 2.
#         capsize (int, optional): Cap size for the error bars. Defaults to 5.
#         capthick (int, optional): Cap thickness for the error bars. Defaults to 2.
#         xlabel (str, optional): Label for the x-axis. Defaults to "x".
#         ylabel (str, optional): Label for the y-axis. Defaults to r'$\log[n(x|\nu) / n(x|0)]$'.
#         xscale (str, optional): Scale of the x-axis. Defaults to "linear".
#         yscale (str, optional): Scale of the y-axis. Defaults to "linear".
#         xlims (tuple, optional): Limits for the x-axis. Defaults to None.
#         ylims (tuple, optional): Limits for the y-axis. Defaults to None.
#         analytic (function, optional): Analytic function to plot. Defaults to None.
#     """
#     fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
#     set_label_font(ax, fontsize)
#     set_tick_font(ax, fontsize - 2)
#     if grid:
#         draw_grid(ax)

#     if zero_line:
#         ax.axhline(y=0, color="black", ls="--", lw=2, alpha=0.5)
        
        
#     if isinstance(xlims, tuple):
#         xgrid = np.linspace(xlims[0], xlims[1], 1000)
#     else:
#         xgrid = np.linspace(bins[0], bins[-1], 1000)

#     if analytic is not None:
#         # check that analytic is a function
#         if not callable(analytic):
#             raise TypeError("analytic must be a function")
#         for nu in nu_list:
#             ax.plot(
#                 xgrid, 
#                 np.log(analytic(xgrid, nu, 0.2)), 
#                 color=palette[np.where(nu==nu_list)[0][0]], 
#                 lw=2, 
#                 ls="-",
#                 alpha=0.5
#             )
        
        
#     bincenters = (bins[1:] + bins[:-1]) / 2
#     binwidths  = bins[1:] - bins[:-1]

#     for nu_iter in range(len(nu_list)):
#         nu     = nu_list[nu_iter]
#         nu_std = nu_list_std[nu_iter]

#         x_ref = feature[(nuisance == nu_std) & (targets == 0)][:, 0]
#         w_ref = weights[(nuisance == nu_std) & (targets == 0)]

#         x_nu = feature[(nuisance == nu_std) & (targets == 1)][:, 0]
#         w_nu = weights[(nuisance == nu_std) & (targets == 1)]

#         histo_ref, _ = np.histogram(x_ref, bins=bins, weights=w_ref)
#         histo_nu, _ = np.histogram(x_nu, bins=bins, weights=w_nu)

#         histo_ref_sq, _ = np.histogram(x_ref, bins=bins, weights=w_ref ** 2)
#         histo_nu_sq, _ = np.histogram(x_nu, bins=bins, weights=w_nu ** 2)

#         ratio = histo_nu * 1.0 / histo_ref
#         ratio_err = ratio * np.sqrt((histo_nu_sq * 1.0 / histo_nu ** 2) + (histo_ref_sq * 1.0 / histo_ref ** 2))

#         log_ratio = np.log(ratio)
#         log_ratio_err = ratio_err / ratio

#         ax.errorbar(
#             x=bincenters,
#             y=log_ratio,
#             yerr=log_ratio_err,
#             xerr=binwidths / 2,
#             ls='',
#             marker=marker,
#             color=palette[nu_iter],
#             markersize=markersize,
#             elinewidth=elinewidth,
#             capsize=capsize,
#             capthick=capthick,
#             label=f'$\\nu$ = {nu}$\sigma$',
#         )

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xscale(xscale)
#     ax.set_yscale(yscale)
    
#     if isinstance(xlims, tuple):
#         ax.set_xlim(xlims)
#     if isinstance(ylims, tuple):
#         ax.set_ylim(ylims)
    
#     ax.legend(fontsize=fontsize - 4, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    
#     plt.show()

#     if return_fig:
#         return fig, ax
    

# def plot_nuisance_learned_ratio(
#     feature,
#     target,
#     model,
#     bins,
#     nu_list,
#     nu_list_std,
#     figsize,
#     fontsize,
#     palette,
#     marker='o',
#     markersize=10,
#     elinewidth=2,
#     capsize=5,
#     capthick=2,
#     xlabel="x",
#     ylabel=r'$\log[n(x|\nu) / n(x|0)]$',
#     xscale="log",
#     yscale="linear",
#     xlims=(1.1e-1, 0.3e2),
#     ylims=None,
#     device="cpu",
#     grid = True,
#     analytic = None,
#     zero_line = True,
#     return_fig=False,
#     constrained=True,
#     ):
#     """
#     Plots the learned ratio of nuisance variations compared to the reference.

#     Args:
#         feature (np.ndarray): The feature array for the test set.
#         target (np.ndarray): The target array for the test set.
#         model (torch.nn.Module): The trained model for calculating learned ratios.
#         bins (np.ndarray): Bin edges for histogram.
#         nu_list (np.ndarray): Array of nuisance parameter values.
#         nu_list_std (np.ndarray): Standardized nuisance parameter values.
#         figsize (tuple): Size of the figure.
#         fontsize (int): Font size for labels.
#         palette (list): List of colors for different nuisance parameters.
#         marker (str, optional): Marker for the error bars. Defaults to 'o'.
#         markersize (int, optional): Marker size for the error bars. Defaults to 10.
#         elinewidth (int, optional): Line width for the error bars. Defaults to 2.
#         capsize (int, optional): Cap size for the error bars. Defaults to 5.
#         capthick (int, optional): Cap thickness for the error bars. Defaults to 2.
#         xlabel (str, optional): Label for the x-axis. Defaults to "x".
#         ylabel (str, optional): Label for the y-axis. Defaults to r'$\log[n(x|\nu) / n(x|0)]$'.
#         xscale (str, optional): Scale of the x-axis. Defaults to "log".
#         xlims (tuple, optional): Limits for the x-axis. Defaults to (1.1e-1, 0.3e2).
#         device (str, optional): Device for model inference. Defaults to "cpu".
#         grid (bool, optional): Whether to draw the grid. Defaults to True.
#         analytic (function, optional): Analytic function to plot. Defaults to None.
#         zero_line (bool, optional): Whether to draw the zero line. Defaults to True.
#     """
#     fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained)
#     set_label_font(ax, fontsize)
#     set_tick_font(ax, fontsize - 2)
#     if grid:
#         draw_grid(ax)
    
#     if zero_line:
#         ax.axhline(y=0, color="black", ls="--", lw=2, alpha=0.5)
    
#     if analytic is not None:
#         if not callable(analytic):
#             raise TypeError("analytic must be a function")
        
#         if isinstance(xlims, tuple):
#             xgrid = np.linspace(xlims[0], xlims[1], 1000)
#         else:
#             xgrid = np.linspace(bins[0], bins[-1], 1000)
        
#         for nu in nu_list:
#             ax.plot(
#                 xgrid, 
#                 np.log(analytic(xgrid, nu, 0.2)), 
#                 color=palette[np.where(nu==nu_list)[0][0]], 
#                 lw=2, 
#                 ls="-",
#                 alpha=0.5
#             )


#     bincenters = (bins[1:] + bins[:-1]) / 2
#     binwidths = bins[1:] - bins[:-1]

#     for nu_iter in range(len(nu_list)):
#         nu = nu_list[nu_iter]
#         nu_std = nu_list_std[nu_iter]

#         # Process reference and nuisance data
#         x_ref, w_ref, x_nu, w_nu = process_nuisance_data(feature, target, nu_std)

#         # Compute and plot binned log-ratio
#         log_ratio, log_ratio_err = compute_log_ratio(x_ref, w_ref, x_nu, w_nu, bins)
#         ax.errorbar(
#             x=bincenters,
#             y=log_ratio,
#             yerr=log_ratio_err,
#             xerr=binwidths / 2,
#             marker=marker,
#             ls='',
#             lw=0,
#             label=f'$\\nu$ = {nu}$\\sigma$ binned',
#             color=palette[nu_iter],
#             markersize=markersize,
#             elinewidth=elinewidth,
#             capsize=capsize,
#             capthick=capthick,
#             alpha=0.25
#         )

#         # Compute and plot learned log-ratio
#         learned_log_ratio = compute_learned_log_ratio_nu(feature, target, model, nu_std, bins, device)
#         ax.plot(bincenters, learned_log_ratio, color=palette[nu_iter], lw=3, ls='--', label=f'$\\nu$ = {nu}$\\sigma$ learned')

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xscale(xscale)
#     ax.set_yscale(yscale)

#     if isinstance(xlims, tuple):
#         ax.set_xlim(xlims)
#     if isinstance(ylims, tuple):
#         ax.set_ylim(ylims)    

#     ax.legend(fontsize=fontsize - 4, ncol=2, bbox_to_anchor=(1.0, 1.0), loc="upper left")

#     plt.show()
    
#     if return_fig:
#         return fig, ax


# def plot_nuisance_learned_binned_difference(
#     feature, 
#     target, 
#     model, 
#     bins, 
#     nu_list, 
#     nu_list_std, 
#     figsize, 
#     fontsize, 
#     palette, 
#     device,
#     xlabel="x",
#     ylabel=r'$\log[n(x|\nu) / n(x|0)]$',
#     xscale="log",
#     yscale="linear",
#     xlims=(1.1e-1, 0.3e2),
#     ylims=None,
#     grid=True,
#     sharex=True,
#     sharey=True,
#     return_fig=False,
#     constrained=True,
#     ):
#     """
#     Plots the normalized differences between learned and binned log-ratios for each nuisance parameter in subplots.

#     Args:
#         feature (np.ndarray): The feature array for the test set.
#         target (np.ndarray): The target array for the test set.
#         model (torch.nn.Module): The trained model for calculating learned ratios.
#         bins (np.ndarray): Bin edges for histogram.
#         nu_list (np.ndarray): Array of nuisance parameter values.
#         nu_list_std (np.ndarray): Standardized nuisance parameter values.
#         figsize (tuple): Size of the figure.
#         fontsize (int): Font size for labels.
#         palette (list): List of colors for different nuisance parameters.
#         device (str): Device for model inference.
#     """
#     fig, axes = plt.subplots(len(nu_list), 1, figsize=figsize, sharex=sharex, sharey=sharey, constrained_layout=constrained)

#     for nu_iter, ax in enumerate(axes):
#         set_tick_font(ax, fontsize - 2)
#         set_label_font(ax, fontsize)
        
#         if grid:
#             draw_grid(ax)
            
#         nu = nu_list[nu_iter]
#         nu_std = nu_list_std[nu_iter]
        
#         # Process reference and nuisance data
#         x_ref, w_ref, x_nu, w_nu = process_nuisance_data(feature, target, nu_std)

#         # Compute binned log-ratio
#         log_ratio, log_ratio_err = compute_log_ratio(x_ref, w_ref, x_nu, w_nu, bins)

#         # Compute learned log-ratio    
#         learned_log_ratio = compute_learned_log_ratio_nu(feature, target, model, nu_std, bins, device)
        
#         bincenters = (bins[1:] + bins[:-1]) / 2
#         y = (learned_log_ratio - log_ratio) / log_ratio_err

#         ax.axhline(y=0, color='black', linestyle='--', lw=2, alpha=0.5)

#         ax.errorbar(
#             x=bincenters,
#             y=y,
#             yerr=np.ones_like(y),  # yerr is 1 as error in normalized difference is 1
#             marker='o',
#             ls='',
#             lw=0,
#             label=f'$\\nu$ = {nu}$\\sigma$',
#             color=palette[nu_iter], 
#             markersize=10,
#             elinewidth=2,
#             capsize=5,
#             capthick=2,
#             alpha=1
#         )
#         ax.legend(fontsize=fontsize - 4, bbox_to_anchor=(1, 1), loc='upper left')
#         ax.set_yscale(yscale)
#         ax.set_xscale(xscale)


#     axes[-1].set_xlabel(xlabel)
#     axes[0].set_ylabel(ylabel)

#     if isinstance(xlims, tuple):
#         axes[0].set_xlim(xlims)
#     if isinstance(ylims, tuple):
#         axes[0].set_ylim(ylims)

#     plt.show()
    
#     if return_fig:
#         return fig, axes
