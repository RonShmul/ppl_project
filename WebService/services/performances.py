import base_handler
import api
import json
import utils
import os.path
SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Performances(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('file')
        performances = api.get_performances(file)
        images = [utils.get_image_string(os.path.join(SOURCE, 'outputs/summary_plot_bar.png')),
                  utils.get_image_string(os.path.join(SOURCE, 'outputs/dependence_plot.png')),
                  utils.get_image_string(os.path.join(SOURCE, 'outputs/summary_plot.png'))]
        if performances:
            self.set_status(200)
            self.write(json.dumps({'performance': performances,
                                   'images': images}))
        else:
            self.set_status(500)
