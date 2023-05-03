from flask import Flask, render_template, request, jsonify
from summarizer import get_article_content, summarize, lsa_summarization

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        model = request.form['model']
        article_content = get_article_content(url)

        if model == 'nltk':
            summary = summarize(article_content)
        elif model == 'lsa':
            summary = lsa_summarization(article_content)

        print(f"Model: {model}")  # Debugging message
        print(f"Summary: {' '.join(summary)}")  # Debugging message

        return jsonify(summary='. '.join(summary) + '.')  # Add periods between sentences and at the end

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
