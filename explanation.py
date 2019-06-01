import shap
import os.path
import utils
ROOT = os.path.abspath(os.path.join(__file__, '../'))


def explain_model(model, X):
    # Visualization for feature importance with SHAP (global)
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    utils.save_picture(os.path.join(ROOT, 'outputs/summary_plot_bar.png'))
    utils.clear_plot()
    shap.dependence_plot("post_length", shap_values, X, show=False)
    utils.save_picture(os.path.join(ROOT, 'outputs/dependence_plot.png'))
    utils.clear_plot()
    shap.summary_plot(shap_values, X, show=False)
    utils.save_picture(os.path.join(ROOT, 'outputs/summary_plot.png'))
    utils.clear_plot()


def explain_class(model,  X):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # visualize the first prediction's explanation
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False,
                    link="logit")
    utils.save_picture(os.path.join(ROOT, 'outputs/force_plot_post.png'))
    utils.clear_plot()
