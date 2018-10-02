import * as d3 from 'd3';
const backend_url = "http://127.0.0.1:5000"
let [resize_timeout_length, resize_timeout] = [66, null];
const [CNN, RL, GAN] = [
    {},
    {
        init_json: `{"relations": [{"label": "bottle", "bbox": [311, 456, 461, 525], "relationships": []}, {"label": "plate", "bbox": [483, 537, 387, 499], "relationships": [{"label": "on", "index": 7}]}, {"bbox": [227, 395, 669, 865], "label": "person", "relationships": [{"label": "behind", "index": 9}]}, {"bbox": [417, 642, 955, 1024], "label": "pot", "relationships": [{"label": "in", "index": 9}]}, {"bbox": [148, 178, 83, 114], "label": "phone", "relationships": [{"label": "has", "index": 6}]}, {"label": "person", "bbox": [171, 372, 444, 639], "relationships": []}, {"label": "person", "bbox": [10, 372, 10, 143], "relationships": []}, {"label": "cup", "bbox": [457, 509, 410, 473], "relationships": []}, {"bbox": [587, 642, 446, 579], "label": "shoes", "relationships": [{"label": "wear", "index": 5}]}, {"label": "tree", "bbox": [1, 416, 795, 1023], "relationships": []}, {"bbox": [321, 479, 2, 60], "label": "chair", "relationships": [{"label": "in", "index": 12}, {"label": "behind", "index": 6}]}, {"bbox": [41, 70, 43, 115], "label": "glasses", "relationships": [{"label": "wear", "index": 6}]}, {"label": "person", "bbox": [147, 674, 50, 359], "relationships": []}, {"bbox": [384, 671, 288, 899], "label": "table", "relationships": [{"label": "at", "index": 12}, {"label": "on", "index": 1}, {"label": "on", "index": 0}, {"label": "behind", "index": 5}, {"label": "under", "index": 8}]}]}`
    },
    {},
];

const margin = parseFloat(getCSSVariable("--svg-margin"));
const graph_size = parseFloat(getCSSVariable("--graph-size"));

function startTheThing() {
    setupCNN();
    setupRL();
    setupGAN();
    window.addEventListener("resize", () => {
        if (!resize_timeout) {
            resize_timeout = setTimeout(() => {
                resize_timeout = null;
                //resizeSVG(CNN);
                //resizeSVG(RL);
                //resizeSVG(GAN);
            }, resize_timeout_length);
        }
    }, false);
}

function setupCNN() {
    doCommonStuff(CNN, "CNN");
    setupSVG(CNN);
    resizeSVG(CNN);
}

function setupRL() {
    doCommonStuff(RL, "RL");
    setupSVG(RL);
    setupCanvasesRL();
    setupGraphRL();
    processAnnotationsRL(JSON.parse(RL.init_json).relations);
    drawBoxesRL();
    updateGraphNodesAndLinksRL();
    RL.button.node().onclick = () => makeGetRequest(
        "/relationships",
        handleRequestRL
    );
}

function setupCanvasesRL() {
    RL.graph_canvas = RL.svg.append("g");
    resizeSVG(RL);
}

function handleRequestRL(request) {
    const json_data = JSON.parse(request.response);
    processAnnotationsRL(json_data.relations);
    RL.img.node().onload = () => {
        resizeSVG(RL);
        drawBoxesRL();
        updateGraphNodesAndLinksRL();
    }
    RL.img.node().src = backend_url + json_data.image_path;
}

function processAnnotationsRL(relations) {
    [RL.nodes, RL.links] = [[], []];
    for (let i = 0; i < relations.length; i++) {
        RL.nodes.push({
            label: relations[i].label,
            ...makeBoxNice(relations[i].bbox),
        });
        for (let j = 0; j < relations[i].relationships.length; j++) {
            RL.links.push({
                label: relations[i].relationships[j].label,
                source: i,
                source_i: i,
                target: relations[i].relationships[j].index,
                target_i: relations[i].relationships[j].index,
            })
        }
    }
}

function makeBoxNice(box) {
    return {
        box: box,
        left: box[2],
        right: box[3],
        up: box[0],
        down: box[1],
        up_left: {x: box[2], y: box[0]},
        up_right: {x: box[3], y: box[0]},
        down_left: {x: box[2], y: box[1]},
        down_right: {x: box[3], y: box[1]},
    };
}

function setupGraphRL() {
    RL.graph_canvas.append("rect")
        .attr("class", "black-white-outline-rect")
        .attr("height", graph_size)
        .attr("width", graph_size);
    RL.graph_canvas.append("rect")
        .attr("class", "black-white-outline-rect other")
        .attr("height", graph_size)
        .attr("width", graph_size);
    RL.link_g = RL.graph_canvas.append("g");
    RL.node_g = RL.graph_canvas.append("g");
    RL.select_g = RL.graph_canvas.append("g");
    const body_strength = parseFloat(getCSSVariable("--sim-body-strength"));
    const link_distance = parseFloat(getCSSVariable("--sim-link-distance"));
    const radial_distance = parseFloat(getCSSVariable("--sim-radial-distance"));

    RL.sim = d3.forceSimulation()
        .force("center", d3.forceCenter(graph_size / 2, graph_size / 2))
        .force("radial", d3.forceRadial(graph_size / 2, graph_size / 2)
            .radius(radial_distance))
        .force("charge", d3.forceManyBody()
            .strength(body_strength))
        .force("link", d3.forceLink()
            .distance(link_distance))
        .on("tick", updateGraphViewRL);
}

function drawBoxesRL() {
    RL.image_canvas.selectAll(".box").remove();
    const rect_enter = RL.image_canvas.selectAll(".box")
        .data(RL.nodes)
        .enter();
    rect_enter.append("rect")
        .attr("class", "box")
        .attr("x", d => RL.scale_width(d.left))
        .attr("y", d => RL.scale_height(d.up))
        .attr("width", d => RL.scale_width(d.right - d.left))
        .attr("height", d => RL.scale_height(d.down - d.up));
    RL.box_rects = rect_enter.append("rect")
        .attr("class", "box other")
        .attr("x", d => RL.scale_width(d.left))
        .attr("y", d => RL.scale_height(d.up))
        .attr("width", d => RL.scale_width(d.right - d.left))
        .attr("height", d => RL.scale_height(d.down - d.up));
}

function updateGraphNodesAndLinksRL() {
    RL.sim.nodes(RL.nodes);
    RL.sim.force("link").links(RL.links);
    updateGraphLinksRL();
    updateGraphNodesRL();
    updateGraphSelectRL();
    RL.sim.alpha(1).restart();
}

function updateGraphNodesRL() {
    RL.node_g.selectAll("g").remove();
    const graph_nodes = RL.node_g.selectAll("g")
        .data(RL.nodes)
        .enter().append("g");
    RL.graph_circles = graph_nodes.append("circle")
        .attr("class", "node");
    graph_nodes.append("text")
        .attr("class", "graph-node-text")
        .attr("transform", translate(8, 3.5))
        .text(d => d.label);
}

function updateGraphLinksRL() {
    RL.link_g.selectAll("g").remove();
    const graph_links = RL.link_g.selectAll("g")
        .data(RL.links)
        .enter().append("g");
    graph_links.append("path")
        .attr("class", "link");
    graph_links.append("circle")
        .attr("class", "link-node");
    graph_links.append("polygon")
        .attr("points", getArrowPoints());
    graph_links.append("text")
        .attr("class", "graph-node-text")
        .attr("transform", translate(8, 3.5))
        .text(d => d.label);
}

function updateGraphSelectRL() {
    RL.select_g.selectAll(".circle-select-node").remove();
    RL.select_g.selectAll(".circle-select-node")
        .data(RL.nodes)
        .enter().append("circle")
            .attr("class", "circle-select-node")
            .on("mouseover", (d, i, nodes) => {
                d3.select(RL.graph_circles.nodes()[i])
                    .attr("class", "node node-highlighted")
                d3.select(RL.box_rects.nodes()[i])
                    .attr("class", "box other box-highlighted");
            }).on("mouseout", (d, i, nodes) => {
                d3.select(RL.graph_circles.nodes()[i])
                    .attr("class", "node")
                d3.select(RL.box_rects.nodes()[i])
                    .attr("class", "box other");
            }).call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded));
    RL.select_g.selectAll(".circle-select-link").remove();
    RL.select_g.selectAll(".circle-select-link")
        .data(RL.links)
        .enter().append("circle")
            .attr("class", "circle-select-link")
            .on("mouseover", (d, i, nodes) => {
                const link_node = d3.select(RL.link_g.selectAll("g").nodes()[i]);
                link_node.select(".link-node")
                    .attr("class", "link-node link-highlight")
                link_node.select(".link")
                    .attr("class", "link link-highlight")
                d3.select(RL.graph_circles.nodes()[d.source_i])
                    .attr("class", "node node-highlighted")
                d3.select(RL.box_rects.nodes()[d.source_i])
                    .attr("class", "box other box-highlighted");
                d3.select(RL.graph_circles.nodes()[d.target_i])
                    .attr("class", "node node-highlighted")
                d3.select(RL.box_rects.nodes()[d.target_i])
                    .attr("class", "box other box-highlighted");
            }).on("mouseout", (d, i, nodes) => {
                const link_node = d3.select(RL.link_g.selectAll("g").nodes()[i]);
                link_node.select(".link-node")
                    .attr("class", "link-node")
                link_node.select(".link")
                    .attr("class", "link")
                d3.select(RL.graph_circles.nodes()[d.source_i])
                    .attr("class", "node")
                d3.select(RL.box_rects.nodes()[d.source_i])
                    .attr("class", "box other");
                d3.select(RL.graph_circles.nodes()[d.target_i])
                    .attr("class", "node")
                d3.select(RL.box_rects.nodes()[d.target_i])
                    .attr("class", "box other");
            })
}

function getArrowPoints() {
    const width = parseFloat(getCSSVariable("--arrow-width"));
    const length = parseFloat(getCSSVariable("--arrow-length"));
    const offset = parseFloat(getCSSVariable("--arrow-offset"));
    return [
        [-width / 2, -length],
        [0, -offset],
        [width / 2, -length],
    ].map(p => p.join(',')).join(' ');
}

function dragStarted(d) {
    if (!d3.event.active) RL.sim.alphaTarget(1).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
}

function dragEnded(d) {
    if (!d3.event.active) RL.sim.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

function updateGraphViewRL() {
    const arrow_offset = parseFloat(getCSSVariable("--arrow-offset"));
    const arrow_length = parseFloat(getCSSVariable("--arrow-length"));
    
    const links_select = RL.select_g.selectAll(".circle-select-link").nodes();
    RL.link_g.selectAll("g")
        .each((d, i, nodes) => {
            const [x1, y1] = [d.source.x, d.source.y];
            const [x2, y2] = [d.target.x, d.target.y];
            const [dx, dy] = [x2 - x1, y2 - y1];
            const dr = Math.sqrt(dx * dx + dy * dy) / 1.2;
            const d_path = `M${x1},${y1}A${dr},${dr},0 0,1 ${x2},${y2}`;
            d3.select(nodes[i]).select("path")
                .attr("d", d_path);
            const path = d3.select(nodes[i]).select("path").node();
            const p_node = path.getPointAtLength(path.getTotalLength() / 2);
            d3.select(nodes[i]).select(".link-node")
                .attr("cx", p_node.x)
                .attr("cy", p_node.y);
            d3.select(nodes[i]).select("text")
                .attr("transform", translate(p_node.x + 8, p_node.y + 3.5))
            const p_arrow_source = path.getPointAtLength(
                arrow_offset
            );
            const p_arrow_target = path.getPointAtLength(
                arrow_offset + arrow_length
            );
            d3.select(nodes[i]).select("polygon")
                .attr("transform", d => ([
                    translate(p_arrow_source.x, p_arrow_source.y),
                    rotate(degreesToPoint(p_arrow_target, p_arrow_source)),
                ].join(' ')));
            d3.select(links_select[i])
                .attr("transform", d => translate(p_node.x, p_node.y));
        });
    RL.node_g.selectAll("g")
        .attr("transform", d => translate(d.x, d.y))
    RL.select_g.selectAll(".circle-select-node")
        .attr("transform", d => translate(d.x, d.y));
}

function setupGAN() {
    doCommonStuff(GAN, "GAN");
    setupSVG(GAN);
    GAN.img.node().onload = () => {
        resizeSVG(GAN);
    }
}

function setupSVG(scope) {
    scope.image_canvas = scope.svg.append("g")
        .attr("transform", translate(margin, margin));
    scope.image_canvas.append("rect")
        .attr("class", "black-white-outline-rect")
    scope.image_canvas.append("rect")
        .attr("class", "black-white-outline-rect other")
}

function resizeSVG(scope) {
    const node = scope.img.node();
    scope.scale_height = d3.scaleLinear()
        .domain([0, node.naturalHeight])
        .range([0, node.height]);
    scope.scale_width = d3.scaleLinear()
        .domain([0, node.naturalWidth])
        .range([0, node.width]);
    scope.height = node.height;
    scope.width = node.width;
    scope.image_canvas.select(".black-white-outline-rect")
        .attr("height", node.height)
        .attr("width", node.width);
    scope.image_canvas.select(".black-white-outline-rect.other")
        .attr("height", node.height)
        .attr("width", node.width);
    if (scope.graph_canvas) {
        scope.svg
            .attr("height", 3 * margin + scope.height + graph_size)
            .attr("width", 2 * margin + scope.width);
        const x = margin + Math.floor((node.width / 2) - (graph_size / 2));
        const y = margin * 2 + node.height
        scope.graph_canvas
            .attr("transform", translate(x, y));
    } else {
        scope.svg
            .attr("height", 2 * margin + scope.height)
            .attr("width", 2 * margin + scope.width);
    }
}

function getCSSVariable(var_name) {
    const style_sheet = document.styleSheets[0].cssRules[0].style;
    return style_sheet.getPropertyValue(var_name);
}

function doCommonStuff(scope, scope_key) {
    scope.container = d3.select(`#${scope_key}-container`);
    scope.img = scope.container.select("img");
    scope.svg = scope.container.select("svg")
        .attr("transform", translate(0, -margin));
    scope.button = scope.container.select("button");
}

function makeGetRequest(url, ok_callback) {
    let request;
    if (window.XMLHttpRequest) {
        request = new XMLHttpRequest();
    } else {
        request = new ActiveXObject("Microsoft.XMLHTTP");
    }
    request.onreadystatechange = e => {
        if (e.target.readyState === 4) {
            if (e.target.status === 200) {
                ok_callback(e.target);
            } else {
                console.error("Non 200 response recieved:", e.target);
            }
        }
    }
    request.open('GET', backend_url + url, true);
    request.send();
}

function translate(x, y) {
    return `translate(${x}, ${y})`;
}

function rotate(degrees) {
    return `rotate(${degrees}, 0, 0)`;
}

function degreesToPoint(a, b) {
    return Math.atan2(a.y - b.y, a.x - b.x) * (180 / Math.PI) + 90;
}

startTheThing();
