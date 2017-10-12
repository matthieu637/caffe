"""
Caffe network visualization: draw the NetParameter protobuffer.


.. note::

    This requires pydot>=1.0.2, which is not included in requirements.txt since
    it requires graphviz and other prerequisites outside the scope of the
    Caffe.
"""

from caffe.proto import caffe_pb2
import re

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}


def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_edge_label(layer):
    """Define edge label based on layer type.
    """

    if layer.type == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layer.type == 'Convolution' or layer.type == 'Deconvolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layer.type == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
#get input/output dimensions
    elif layer.type == 'MemoryData':
        edge_label = str(layer.memory_data_param.height)
    elif layer.type == 'Slice':
        edge_label = str(layer.slice_param.slice_point[0])
    else:
        edge_label = '""'

    return edge_label


def get_layer_label(layer, rankdir):
    """Define node label based on layer type.

    Parameters
    ----------
    layer : ?
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.

    Returns
    -------
    string :
        A label for the current layer
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = '\\n'

    if layer.type == 'Convolution' or layer.type == 'Deconvolution':
        # Outer double quotes needed or else colon characters don't parse
        # properly
        node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      layer.type,
                      separator,
                      layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1,
                      separator,
                      layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1,
                      separator,
                      layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0)
    elif layer.type == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      pooling_types_dict[layer.pooling_param.pool],
                      layer.type,
                      separator,
                      layer.pooling_param.kernel_size,
                      separator,
                      layer.pooling_param.stride,
                      separator,
                      layer.pooling_param.pad)
    else:
        node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
    return node_label


def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type.
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


def get_pydot_graph(caffe_net, rankdir, label_edges=True, phase=None):
    """Create a data structure which represents the `caffe_net`.

    Parameters
    ----------
    caffe_net : object
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    label_edges : boolean, optional
        Label the edges (default is True).
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)

    Returns
    -------
    pydot graph object
    """
    pydot_graph = pydot.Dot(caffe_net.name if caffe_net.name else 'Net',
                            graph_type='digraph',
                            rankdir=rankdir)
    pydot_nodes = {}
    pydot_edges = []
    slices={}
    inputs = {}
    for layer in caffe_net.layer:
#Scale and Silence are only used internally
        if layer.type == 'Scale' or layer.type == 'Silence' :
            continue
        if phase is not None:
            included = False
            if len(layer.include) == 0:
                included = True
            if len(layer.include) > 0 and len(layer.exclude) > 0:
              raise ValueError('layer ' + layer.name + ' has both include '
                              'and exclude specified.')
            for layer_phase in layer.include:
                included = included or layer_phase.phase == phase
            for layer_phase in layer.exclude:
                included = included and not layer_phase.phase == phase
            if not included:
                continue
        node_label = get_layer_label(layer, rankdir)
        node_name = "%s_%s" % (layer.name, layer.type)
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
           layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            pydot_nodes[node_name] = pydot.Node(node_label,
                                                **NEURON_LAYER_STYLE)
        else:
            layer_style = LAYER_STYLE_DEFAULT
            layer_style['fillcolor'] = choose_color_by_layertype(layer.type)
            pydot_nodes[node_name] = pydot.Node(node_label, **layer_style)
        for bottom_blob in layer.bottom:
            pydot_nodes[bottom_blob + '_blob'] = pydot.Node('%s' % bottom_blob,
                                                            **BLOB_STYLE)
            edge_label = '""'
            if layer.type == "Slice":
                edge_label = inputs[bottom_blob]
                inputs[node_name] = edge_label
            elif bottom_blob in inputs.keys():
                edge_label = inputs[bottom_blob]
                inputs[bottom_blob] = '""'
                
            pydot_edges.append({'src': bottom_blob + '_blob',
                                'dst': node_name,
                                'label': edge_label})

        for top_blob in layer.top:
            pydot_nodes[top_blob + '_blob'] = pydot.Node('%s' % (top_blob))
            if label_edges:
                edge_label = get_edge_label(layer)
                if layer.type == "Slice":
                    if node_name in slices.keys():
                        edge_label = str(int(inputs[node_name]) - int(slices[node_name]))
                    else:
                        slices[node_name] = edge_label
                    inputs[top_blob] = edge_label
                elif layer.type == "MemoryData":
                    inputs[top_blob] = edge_label
            else:
                edge_label = '""'
            pydot_edges.append({'src': node_name,
                                'dst': top_blob + '_blob',
                                'label': edge_label})

#remove dummy and MemoryData node
    ignored_blob=[]
    dummy_pattern = re.compile('dummy*')
    memory_pattern = re.compile('.*MemoryData.*')
    slice_pattern = re.compile('.*Slice.*')
    reshape_pattern = re.compile('.*Reshape.*')
    concat_pattern = re.compile('.*Concat.*')
    states_pattern = re.compile('.*states.*')
    actions_pattern = re.compile('.*actions.*')
    states_pattern_d = re.compile('.*states[.].*')
    actions_pattern_d = re.compile('.*actions[.].*')
    states_actions_pattern = re.compile('.*states_actions.*')
    developmental=False
    containsSeveralStates=False
    for node in pydot_nodes.values():
        if states_pattern_d.match(node.get_name()) or actions_pattern_d.match(node.get_name()):
            developmental=True
        if states_pattern_d.match(node.get_name()):
            containsSeveralStates=True
    # Now, add the nodes and edges to the graph.
    for node in pydot_nodes.values():
        if dummy_pattern.match(node.get_name()) or memory_pattern.match(node.get_name()) or node.get_name() == 'loss':
            continue
        if node.get_shape() == 'octagon' and not states_pattern.match(node.get_name()) and not actions_pattern.match(node.get_name()):
            ignored_blob.append(node.get_name())
            continue
        if developmental and (node.get_name() == "actions" or (node.get_name() == "states" and containsSeveralStates) or slice_pattern.match(node.get_name()) ):
            ignored_blob.append(node.get_name())
            continue
        if developmental and actions_pattern_d.match(node.get_name()) and not is_labeled(pydot_edges, pydot_nodes, node.get_name()):
            ignored_blob.append(node.get_name())
            continue
        if states_actions_pattern.match(node.get_name()) or reshape_pattern.match(node.get_name()) or concat_pattern.match(node.get_name()):
            ignored_blob.append(node.get_name())
            continue
        if states_pattern.match(node.get_name()) or actions_pattern.match(node.get_name()) :
            node.set_shape('ellipse')
            node.set_fillcolor('white')
        pydot_graph.add_node(node)

#direct cycle
    while True:
        no_modif=True
        compte={}
        for i in range(len(pydot_edges)-1):
            if pydot_edges[i]['src'] == pydot_edges[i+1]['dst'] and pydot_edges[i+1]['src'] == pydot_edges[i]['dst']:
                if pydot_edges[i]['src'] in compte.keys():
                    compte[pydot_edges[i]['src']] += 1
                else:
                    compte[pydot_edges[i]['src']] = 0
        i=0
        while i < len(pydot_edges)-1:
            if pydot_edges[i]['src'] == pydot_edges[i+1]['dst'] and pydot_edges[i+1]['src'] == pydot_edges[i]['dst']:
                j=i+2
                firstTime=True
                blob_save=pydot_edges[i+1]['src']
                while j < len(pydot_edges):
                    if pydot_edges[j]['src'] == pydot_edges[i]['src']:
                        if firstTime:
                            pydot_edges[j]['src'] = pydot_edges[i+1]['src']
                            pydot_edges.pop(i+1)
                            j=i+1
                            firstTime=False
                            if compte[pydot_edges[i]['src']] == 1:
                                break
                        else:
                            pydot_edges[j]['src'] = blob_save
                        no_modif=False
                    j+=1
            if no_modif==False:
                break
            i+=1
        if no_modif:
            break

#3 edges cycles
    while True:
        no_modif=True
        for i in range(len(pydot_edges)-2):
            if pydot_edges[i]['src'] == pydot_edges[i+2]['dst'] and pydot_edges[i+1]['src'] == pydot_edges[i]['dst'] and pydot_edges[i+2]['src'] == pydot_edges[i+1]['dst']:
                j=i+3
                firstTime=True
                blob_save=pydot_edges[i+2]['src']
                while j < len(pydot_edges):
                    if pydot_edges[j]['src'] == pydot_edges[i]['src']:
                        if firstTime:
                            pydot_edges[j]['src'] = pydot_edges[i+2]['src']
                            pydot_edges.pop(i+2)
                            j=i+2
                            firstTime=False
                        else:
                            pydot_edges[j]['src'] = blob_save
                        no_modif=False
                    j+=1
            if no_modif==False:
                break
        if no_modif:
            break
#ignored blob
    while True:
        no_modif=True
        for i in range(len(pydot_edges)-2):
            if pydot_nodes[pydot_edges[i]['dst']].get_name() in ignored_blob :
                j=0
                first=True
                blob_prob = pydot_edges[i]['dst']
                while j < len(pydot_edges):
                    if pydot_nodes[pydot_edges[j]['src']].get_name() == pydot_nodes[blob_prob].get_name():
                        if first:
                            pydot_edges[i]['dst'] = pydot_edges[j]['dst']
                            blob_save=pydot_edges[j]['dst']
                            pydot_edges.pop(j)
                            j=-1
                            first=False
                        else:
                            pydot_edges[j]['src'] = pydot_edges[i]['src']
                            pydot_edges[j]['label'] = pydot_edges[i]['label']
                        no_modif=False
                    j+=1
                if not first:
                    for j in range(len(pydot_edges)):
                        if pydot_nodes[pydot_edges[j]['dst']].get_name() == pydot_nodes[blob_prob].get_name():
                            pydot_edges[j]['dst'] = blob_save
                            no_modif=False

            if no_modif==False:
                break
        if no_modif:
            break
   

    for edge in pydot_edges:
        src_name = pydot_nodes[edge['src']].get_name()
        dst_name = pydot_nodes[edge['dst']].get_name()
        if dummy_pattern.match(src_name) or dummy_pattern.match(dst_name) or src_name == 'target' or dst_name == 'loss':
            continue
        if memory_pattern.match(src_name):
            continue
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge['src']],
                       pydot_nodes[edge['dst']],
                       label=edge['label']))
    return (pydot_graph,developmental)

def is_labeled(pydot_edges, pydot_nodes, dst):
    for edge in pydot_edges:
        if pydot_nodes[edge['dst']].get_name() == dst and edge['label'] != '""':
            return True
    return False

def node_from_edge(nodes, name):
    for n in nodes:
        if n.get_name() == name:
            return n
    return None

def compact(graph, rankdir, cluster):
    pydot_graph = pydot.Dot(graph.get_name(),
                        graph_type=graph.get_type(),
                        rankdir=rankdir)
    #pydot_graph.set('compound', 'true')
#    pydot_graph.set_edge_defaults(**{'tailclip':'false'});
    
    if cluster:
        c1=pydot.Subgraph('task0')
        c2=pydot.Subgraph('task1')
        c1.set('label', 'task0')
        c1.set('style', 'filled')
        c1.set('color', 'blue')
        pydot_graph.add_subgraph(c1)
        pydot_graph.add_subgraph(c2)
    
    final_edge = []
    ignored_node = []
    ignored_edge = []
    edge_src = {}
    
    ip_pattern = re.compile('.*InnerProduct.*')
    func_pattern = re.compile('.*func.*')
    bn_pattern = re.compile('.*BatchNorm.*')
    
    for edge1 in graph.get_edge_list():
        if ip_pattern.match(edge1.get_source()) and func_pattern.match(edge1.get_destination()):
            base = node_from_edge(graph.get_nodes(), edge1.get_source())
            newn = pydot.Node(base.get_name())
            name = re.search('.*[(](.*)[)]', base.get_name()).group(1)
            newn.set('shape', 'none')
            label = "<<table border='0' cellborder='0' cellspacing='0'>"
            label += "<tr><td width='100' border='0' cellspacing='0' >"+edge1.get('label')+"</td></tr>"
            label += "<tr><td port='pp' width='100' border='1' cellspacing='0' bgcolor='"+base.get('fillcolor')+"' >"+name+"</td></tr>"
            second = node_from_edge(graph.get_nodes(), edge1.get_destination())
            name = re.search('.*[(](.*)[)]', second.get_name()).group(1)
            label += "<tr><td width='100' border='1' cellspacing='0' bgcolor='"+second.get('fillcolor')+"' >"+name+"</td></tr>"
            edge_src[edge1.get_destination()] = edge1.get_source()+':pp'
            e2=None
            for edge2 in graph.get_edge_list():
                if edge1.get_destination() == edge2.get_source() and bn_pattern.match(edge2.get_destination()):
                    e2=edge2
                    break
            if e2 != None:
                third = node_from_edge(graph.get_nodes(), edge2.get_destination())
                name = re.search('.*[(](.*)[)]', third.get_name()).group(1)
                ignored_node.append(third.get_name())
                label += "<tr><td width='100' border='1' cellspacing='0' bgcolor='"+third.get('fillcolor')+"' >"+name+"</td></tr>"
                del edge_src[edge1.get_destination()]
                edge_src[edge2.get_destination()] = edge1.get_source()+':pp'
                ignored_edge.append((edge2.get_source(), edge2.get_destination()))
            label += "</table>>"
            newn.set('label', label)
            if cluster:
                if 'task0' in newn.get_name():
                    c1.add_node(newn)
                else:
                    c2.add_node(newn)
            else:
                pydot_graph.add_node(newn)
            ignored_node.append(base.get_name())
            ignored_node.append(second.get_name())
        else :
            final_edge.append(edge1)

    for node in graph.get_nodes():
        if node.get_name() not in ignored_node:
            reg = re.search('.*[(](.*)[)]', node.get_name())
            if reg:
                name = reg.group(1)
                node.set('label', name)
                if name == 'InnerProduct':
                    newn = pydot.Node(node.get_name())
                    for edge in final_edge:
                        if edge.get_source() == node.get_name():
                            siz=edge.get('label')
                            edge.set('label', '""')
                    label = "<<table border='0' cellborder='0' cellspacing='0'>"
                    label += "<tr><td width='100' border='0' cellspacing='0' >"+siz+"</td></tr>"
                    label += "<tr><td width='100' port='pp' border='1' cellspacing='0' bgcolor='"+node.get('fillcolor')+"' >"+name+"</td></tr>"
                    label += "</table>>"
                    newn.set('label', label)
                    newn.set('shape', 'none')
                    node = newn
            
            if cluster:
                if 'task0' in node.get_name():
                    c1.add_node(node)
                else:
                    c2.add_node(node)
            else:
                pydot_graph.add_node(node)

    for edge in final_edge:
        if (edge.get_source(), edge.get_destination()) in ignored_edge:
            continue
        desti = edge.get_destination()
        sourc = edge.get_source()
        node_list=pydot_graph.get_nodes()
        if cluster:
            node_list+=c1.get_nodes()
            node_list+=c2.get_nodes()
        n_dst=node_from_edge(node_list, edge.get_destination())
        n_src=node_from_edge(node_list, edge.get_source())
        if n_dst != None and n_dst.get('label') != None and "port='pp'" in n_dst.get('label'):
            desti=desti+':pp'
        if n_src != None and n_src.get('label') != None and "port='pp'" in n_src.get('label'):
            sourc=sourc+':pp'
        weight=1
        style='solid'
        if cluster:
            if ('task0' in edge.get_source() and 'task1' in edge.get_destination()) or ('task1' in edge.get_source() and 'task0' in edge.get_destination()):
                weight=0
                style='dashed'
            elif (edge.get_source()=="states" and 'task1' in edge.get_destination()):
                weight=0
        if edge.get_source() in edge_src.keys():
            pydot_graph.add_edge(
                  pydot.Edge(src=edge_src[edge.get_source()],
                  dst=desti,
                  label=edge.get('label'), weight=weight, style=style))
        else:
            pydot_graph.add_edge(                  
                  pydot.Edge(src=sourc,
                  dst=desti,
                  label=edge.get('label'), weight=weight, style=style))
    
    return pydot_graph

def draw_net(caffe_net, rankdir, ext='png', phase=None):
    """Draws a caffe net and returns the image string encoded using the given
    extension.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    ext : string, optional
        The image extension (the default is 'png').
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)

    Returns
    -------
    string :
        Postscript representation of the graph.
    """
    (first,dev)=get_pydot_graph(caffe_net, rankdir, phase=phase)
    second=compact(first, rankdir, dev)
    return second.create(format=ext)


def draw_net_to_file(caffe_net, filename, rankdir='LR', phase=None):
    """Draws a caffe net, and saves it to file using the format given as the
    file extension. Use '.raw' to output raw text that you can manually feed
    to graphviz to draw graphs.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    filename : string
        The path to a file where the networks visualization will be stored.
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    """
    ext = filename[filename.rfind('.')+1:]
    with open(filename, 'wb') as fid:
        fid.write(draw_net(caffe_net, rankdir, ext, phase))
