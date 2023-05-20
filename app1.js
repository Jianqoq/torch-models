// cytoscape.use(dagre);
const dpi = 96;
function convertToRGB(colorString) {
    const colorValues = colorString.split(" ").map((value) => Math.floor(parseFloat(value) * 255));
    return `rgb(${colorValues[0]}, ${colorValues[1]}, ${colorValues[2]})`;
  }

function replace_string(string) {
    const replacedStr = string.replace(/<BR\/>/gi, "\n");
    // const replacedStr = string.replace(/<BR\/>/gi, " ").replace("\n", " ");
    return replacedStr;
}

  
fetch("wo.json").then((response)=> {
    if (response.ok) {
        return response.json();
    } else {
        throw new Error("Failed in fetching data");
    }
}).then((jsonData) => {
    let parsedData = jsonData;
    console.log(parsedData);
    let edges = parsedData.edges.map((edge) => {
        return {
            data: {
                id: `edge-${edge.tail}-${edge.head}`,
                source: edge.tail,
                target: edge.head,
                label: replace_string(edge.label),
            },
        };
    });

    let nodes = parsedData.objects.map((obj, index) => {
        let shape = obj.shape === "box" ? "roundrectangle" : obj.shape;
        return {
          data: {
            id: obj._gvid,
            label: replace_string(obj.label),
            fillcolor: convertToRGB(obj.fillcolor),
            width: parseFloat(obj.width) * dpi,
            height: parseFloat(obj.height) * dpi,
            shape: shape,
            style: obj.style,
            position: {
              x: parseFloat(obj.pos.split(",")[0]),
              y: parseFloat(obj.pos.split(",")[1]),
            },
          },
        };
      });
    
      let cy = cytoscape({
        container: document.getElementById("cy"), // 在HTML页面�为Cytoscape图创建一��器元�
        elements: nodes.concat(edges),
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(fillcolor)",
              label: "data(label)",
              shape: "data(shape)",
              content: "data(label)",
              "width": "data(width)",
              "height": "data(height)",
              "font-family": "Arial, sans-serif", // 指定节点标�的字�
              "font-size": "14px", // 指定节点标�的字体大�
              "text-wrap": "wrap", // ��文本��
              "border-width": 2,      // 边��度
              "border-color": "black"   // 边��色
            },
          },
          {
            selector: "edge",
            style: {
              label: "data(label)",
              "line-color": "#ccc",
              "target-arrow-color": "#ccc",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              "font-family": "Arial, sans-serif", // 指定节点标�的字�
              "font-size": "14px", // 指定节点标�的字体大�
              "text-wrap": "wrap",
              "text-opacity": 0,
            },
        },
        {
            selector: "node[label]",
            style: {
              "label": "data(label)",
              "text-halign": "center",
              "text-valign": "center",
            },
          },

          {
            selector: "edge.clicked",
            style: {
              "text-opacity": 1,
            },
          },

        ],
        layout: {
          name: "dagre",
          nodeSep: 30, // 节点之间的水平间�
          rankSep: 30, // 排名之间的垂直间�
        },
        wheelSensitivity: 0.03,
        minZoom: 0, // 设置�小缩放比�
        maxZoom: 6, // 设置�大缩放比�
      });

      const cxtMenuDefaults = {
        menuRadius: 100,
        selector: "node",
        commands: [
          {
            content: "添加节点",
            select: function (ele) {
              // 添加节点的代�
            },
          },
          {
            content: "删除节点",
            select: function (ele) {
              // 删除节点的代�
            },
          },
        ],
      };

      cy.nodes().forEach((node) => {
        let label = node.data("label");
        let labelproperty = getTextWidthHeight(label, "14px Arial, sans-serif");
        let extraWidth = 20; // 加上�些��空�
        let extraHeight = 20; // 加上�些��空�
        node.style({ "overlay-opacity": 0 });
        // 设置节点的自适应宽度
        node.style("width", labelproperty[0] + extraWidth);
        node.style("height", labelproperty[1] + extraHeight);
      });
      

      // 获取文本的��
      function getTextWidthHeight(text, font) {
        let canvas = document.createElement("canvas");
        let context = canvas.getContext("2d");
        context.font = font;
        let metrics = context.measureText(text);
        return [metrics.width, metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent];
      }

      cy.on("click", "edge", function (event) {
        event.target.toggleClass("clicked");
      });

      let mouseDownNode;
      cy.on("mousedown", "node", function (event) {
        const targetNode = event.target;
        mouseDownNode = targetNode;
      
        targetNode
          .animate({
            style: { "background-color": "#D3D3D3",
                      "color": "white" }, // 您可以�择��的�色
          }, {
            duration: 50, // 动画持续时间（以��为单位�
          });
      });

      cy.on("mouseup", function (event) {
        if (mouseDownNode) {
          mouseDownNode
            .animate({
              style: { "background-color": mouseDownNode.data("fillcolor"),
                        "color": "black"
             }, // ��节点数��� fillcolor
            }, {
              duration: 50, // 动画持续时间（以��为单位�
            });
      
          mouseDownNode = null;
        }
      });

      cy.on("tap", "node, edge", function (event) {
        let node = event.target;
        let label = node.data("label");
      
        event.target.toggleClass("tapped");
        copyToClipboard(label);
      
        console.log("已�制标���" + label);

      });
      
      // 复制文本到剪贴板
      function copyToClipboard(text) {
        let string = text.split("\n")[0];
        navigator.clipboard.writeText(string)
          .catch((error) => {
            console.error("复制失败:", error);
          });
      }
      

})
