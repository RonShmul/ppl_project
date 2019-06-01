import tornado.ioloop
import tornado.web
from services import train
from services import predict
from services import performances


def make_app(debug):
    return tornado.web.Application(
        [
            ('/train', train.Train),
            ('/get_classification', predict.Predict),
            ('/get_performance', performances.Performances),
        ],
        debug=debug
    )


def main():
    app = make_app(True)
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


main()
