from flask import Flask, render_template, request, jsonify
import base64
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


import io
def build_graph(x_coordinates, y_coordinates):
    img = io.BytesIO()
    plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


from gensim.models import word2vec
model = word2vec.Word2Vec.load("word2vec_crm2019.model")


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', user_image = "/static/cloud1.png")


@app.route('/process', methods=['POST'])
def process():
    try:
        name = request.form['name']    
        W = pd.DataFrame(model.most_similar([name],topn=25))
        img = io.BytesIO()
        plt.figure(figsize=(10,12))
        plt.title(name)
        plt.barh(W[0][::-1],W[1][::-1],0.6)
        plt.grid(True)
        ax = plt.subplot(111)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        ret = 'data:image/png;base64,{}'.format(graph_url)
        return jsonify({'img1': ret})
    except:
        return jsonify({'error' : 'error', 'img1': 'error'})
    
    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5005, debug = True, ssl_context=('server.crt', 'server.key'))
    app.run(host='0.0.0.0', port=5005, debug = True)
#    ngrok http 5005






