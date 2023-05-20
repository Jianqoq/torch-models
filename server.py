import http.server
import socketserver

PORT = 8000
DIRECTORY = r"C:\Users\123\PycharmProjects\ML-library\layers"

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server(file_name):
    with open(rf"{DIRECTORY}\app1.js", "r", encoding="gbk", errors="ignore") as fp:
        data = fp.read()
        filename = data.split("fetch(")[1].split(")")[0]
        data = data.replace(f"fetch({filename})", f"fetch(\"{file_name}\")")
    with open("app1.js", "w", encoding="gbk", errors="ignore") as fp:
        fp.write(data)
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print("服务器已启动，访问地址: http://localhost:" + str(PORT))
        httpd.serve_forever()


if __name__ == "__main__":
    start_server("data1.json")