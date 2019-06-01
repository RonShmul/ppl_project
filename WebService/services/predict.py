import base_handler
import api
import os.path
import utils

SOURCE = os.path.abspath(os.path.join(__file__, '../../../'))


class Predict(base_handler.BaseHandler):
    def post(self):
        post = self.body_argument('post')
        explain = self.body_argument('explainability')

        data = api.predict(post, explain)
        if 'error' in data.keys():
            self.set_status(404)
            self.write(data)
            return

        data['explain'] = utils.get_image_string(os.path.join(SOURCE, 'outputs/force_plot_post.png'))
        self.set_status(200)
        self.write(data)


