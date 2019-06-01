import base_handler
import api


class Train(base_handler.BaseHandler):
    def post(self):
        file = self.body_argument('file')
        api.train_file(file)
        self.set_status(200)

