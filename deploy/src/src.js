import * as d3 from 'd3';
import * as tf from '@tensorflow/tfjs';

let objects, predicates, annotations, model;
let image_canvas, image_height, image_width;
let simulation_canvas, simulation, nodes, links;
let [simulation_width, simulation_height] = [350, 350];
const margin = 10;

const promises = [d3.json('./objects.json'),
                  d3.json('./predicates.json'),
                  d3.json('./annotations.json'),
                  //tf.loadModel('./vgg16/model.json'/*'./densenet121/model.json'*/),
                 ];

Promise.all(promises)
    .then(allLoaded)
    .catch(error => console.log(error));

function allLoaded(promises) {
    [objects, predicates, annotations/*, model*/] = promises;
    annotations = annotations.map(makeAnnotationNice);
    createCanvases();
    let points = image_canvas.selectAll('circle').data(annotations);
    points.enter().call(drawBox, 'object');
    points.enter().call(drawBox, 'subject');
    points.enter().each(drawConnect);
    createSimulation();
}

function makeAnnotationNice(annotation) {
    return {
        predicate: {
            label: predicates[annotation.predicate],
            label_i: annotation.predicate,
        },
        object: {
            label: objects[annotation.object.category],
            label_i: annotation.object.category,
            ...makeBBoxNice(annotation.object.bbox),
        }, 
        subject: {
            label: objects[annotation.subject.category],
            label_i: annotation.subject.category,
            ...makeBBoxNice(annotation.subject.bbox),
        },
    };
}

function makeBBoxNice(bbox) {
    return {
        bbox: bbox,
        left: bbox[2],
        right: bbox[3],
        up: bbox[0],
        down: bbox[1],
        up_left: {x: bbox[2], y: bbox[0]},
        up_right: {x: bbox[3], y: bbox[0]},
        down_left: {x: bbox[2], y: bbox[1]},
        down_right: {x: bbox[3], y: bbox[1]},
    };
}

function createCanvases() {
    const img_container = d3.select('#image-container');
    const img = img_container.select('img').node();
    [image_height, image_width] = [img.naturalHeight, img.naturalWidth];
    const canvas = img_container.append('svg')
        .attr('transform', 'translate(' + -margin + ',' + -margin + ')')
        .attr('height', margin * 2 + image_height + simulation_height)
        .attr('width', margin * 3 + image_width + simulation_width)
    image_canvas = canvas.append('g')
        .attr('transform', 'translate(' + margin + ',' + margin + ')');
    simulation_canvas = canvas.append('g')
        .attr('transform', 'translate(' + (margin * 2 + image_width) + ',' + margin + ')');
}

function drawBox(annotations, key) {
    annotations.append('rect')
        .attr('class', 'bbox ' + key)
        .attr('x', d => d[key].left)
        .attr('y', d => d[key].up)
        .attr('width', d => d[key].right - d[key].left)
        .attr('height', d => d[key].down - d[key].up);
}

function drawConnect(annotaion) {
    const [object_corner, subject_corner] = closestCornerPointPair(annotaion, true);
    d3.select(this).append('line')
        .attr('class', 'connection')
        .attr('x1', object_corner.x)
        .attr('y1', object_corner.y)
        .attr('x2', subject_corner.x)
        .attr('y2', subject_corner.y);
}

function closestCornerPointPair(annotation, offset_radius) {
    const corner_keys = ['up_left', 'up_right', 'down_left', 'down_right'];
    let min_dist = Infinity;
    let min_lhs_key, min_rhs_key;
    for (const lhs_key of corner_keys) {
        for (const rhs_key of corner_keys) {
            const [lhs_p, rhs_p] = [annotation.object[lhs_key], annotation.subject[rhs_key]];
            const dist = Math.sqrt(Math.pow(lhs_p.x - rhs_p.x, 2) + Math.pow(lhs_p.y - rhs_p.y, 2));
            if (dist <= min_dist) {
                [min_dist, min_lhs_key, min_rhs_key] = [dist, lhs_key, rhs_key];
            }
        }
    }
    const [y_object, x_object] = min_lhs_key.split('_');
    const [y_subject, x_subject] = min_rhs_key.split('_');
    let radius = 0;
    if (offset_radius) {
        const style = document.styleSheets[0].cssRules[0].style;
        radius = parseFloat(style.getPropertyValue('--radius')) / 2;
    }
    const offsets = {left: radius, right: -radius, up: radius, down: -radius};
    return [
        {
            x: annotation.object[x_object] + offsets[x_object],
            y: annotation.object[y_object] + offsets[y_object]
        }, {
            x: annotation.subject[x_subject] + offsets[x_subject],
            y: annotation.subject[y_subject] + offsets[y_subject]
        },
    ];
}

function createSimulation() {
    nodes = d3.map()
    for (const annotation of annotations) {
        nodes.set(annotation.object.bbox.join('-'), annotation.object.label);
        nodes.set(annotation.subject.bbox.join('-'), annotation.subject.label);
    }
    nodes = nodes.entries().map(x => ({id: x.key, label: x.value}))
    links = annotations.map(annotation => ({
        source: annotation.object.bbox.join('-'),
        target: annotation.subject.bbox.join('-'),
        label: annotation.label,
    }))

    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink().distance(40).id(d => d.id))
        .force('charge', d3.forceManyBody().distanceMax(125).strength(-50))
        .force('center', d3.forceCenter(simulation_width / 2, simulation_height / 2))
        .on('tick', tickSimulation);

    simulation.force('link').links(links);

    links = simulation_canvas.selectAll('line')
        .data(links).enter().append('line')
            .attr('class', 'link')

    nodes = simulation_canvas.selectAll('circle')
        .data(nodes).enter().append('circle')
            .attr('class', 'node')
}

function tickSimulation() {
    links
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
    nodes
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
}
