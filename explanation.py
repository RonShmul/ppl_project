import shap
import matplotlib.pyplot as pl


def explain_model(model, X, folder_name):
    # Visualization for feature importance with SHAP (global)
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    pl.savefig(folder_name + r'/summary_plot_bar.png')
    pl.clf()
    shap.dependence_plot("post_length", shap_values, X, show=False)
    pl.savefig(folder_name + r'/dependence_plot.png')
    pl.clf()
    shap.summary_plot(shap_values, X, show=False)
    pl.savefig(folder_name + r'/summary_plot.png')
    pl.clf()
    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    pl.savefig(folder_name + r'/force_plot_0.png')
    pl.clf()
    shap.force_plot(explainer.expected_value, shap_values[-1, :], X.iloc[-1, :], matplotlib=True, show=False,
                    link="logit")
    pl.savefig(folder_name + r'/force_plot_length.png')
    pl.clf()


def explain_class(model, post,  X, folder_name):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    pl.savefig(folder_name + r'/force_plot_0.png')
    pl.clf()
    shap.force_plot(explainer.expected_value, shap_values[-1, :], X.iloc[-1, :], matplotlib=True, show=False,
                    link="logit")
    pl.savefig(folder_name + r'/force_plot_length.png')
    pl.clf()
