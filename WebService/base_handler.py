import json
import tornado.web
import tornado_cors


class BaseHandler(tornado_cors.CorsMixin, tornado.web.RequestHandler):
    CORS_ORIGIN = '*'
    CORS_HEADERS = 'Content-Type, Authorization'
    CORS_METHODS = 'POST, GET'
    CORS_CREDENTIALS = True
    CORS_MAX_AGE = 21600
    CORS_EXPOSE_HEADERS = 'Location, X-WP-TotalPages'

    def write_error(self, status_code, **kwargs):
        http_error = kwargs['exc_info'][1]
        error_log = ""
        if hasattr(http_error, 'log_message'):
            error_log = http_error.log_message
        self.set_status(status_code)
        self.write({'code': status_code, 'message': error_log})

    def prepare(self):
        if len(self.request.body) > 0:
            self.request.json = json.loads(self.request.body)

    def body_argument(self, key):
        if key not in self.request.json:
            raise tornado.web.HTTPError(400, 'argument missing from request')
        return self.request.json[key]
