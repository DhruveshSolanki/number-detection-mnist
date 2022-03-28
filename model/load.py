import keras.models
from tensorflow import compat

compat.v1.disable_eager_execution()


def init():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model/model.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss,accuracy = model.evaluate(X_test,y_test)
    # print('loss:', loss)
    # print('accuracy:', accuracy)
    return model
