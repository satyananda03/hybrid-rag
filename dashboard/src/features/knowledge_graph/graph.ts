import { MultiUndirectedGraph } from 'graphology'
import { resolveNodeColor } from './graphColor'
import type { KnowledgeGraphResponse } from './types'
import type { EdgeDisplayData, NodeDisplayData, RawEdge, RawGraph, RawNode } from './store'

// Matches graph_ui/src/lib/constants.ts (minNodeSize/maxNodeSize) exactly so
// node sizing reads the same as localhost:8000/webui.
const MIN_NODE_SIZE = 4
const MAX_NODE_SIZE = 20
// Every node gets this border (the node-border program reads `borderColor`);
// matches graph_ui's NODE_BORDER_COLOR.
const NODE_BORDER_COLOR = '#000000'

export function hashPosition(id: string): { x: number; y: number } {
  let left = 0xdeadbeef ^ id.length
  let right = 0x41c6ce57 ^ id.length
  for (let index = 0; index < id.length; index += 1) {
    const char = id.charCodeAt(index)
    left = Math.imul(left ^ char, 2654435761)
    right = Math.imul(right ^ char, 1597334677)
  }
  return { x: (left >>> 0) / 4294967296, y: (right >>> 0) / 4294967296 }
}

function normalizeNode(node: RawNode): RawNode {
  return {
    ...node,
    labels: Array.isArray(node.labels) ? node.labels : [],
    properties: node.properties ?? {},
    degree: 0,
  }
}

function normalizeEdge(edge: RawEdge): RawEdge {
  return { ...edge, properties: edge.properties ?? {} }
}

export function graphFromResponse(response: KnowledgeGraphResponse): RawGraph {
  const nodesById = new Map<string, RawNode>()
  for (const candidate of response.nodes ?? []) {
    if (!candidate?.id || nodesById.has(candidate.id)) continue
    nodesById.set(candidate.id, normalizeNode(candidate as RawNode))
  }
  const edges = (response.edges ?? [])
    .map((edge) => normalizeEdge(edge as RawEdge))
    .filter((edge) => nodesById.has(edge.source) && nodesById.has(edge.target))
  for (const edge of edges) {
    const source = nodesById.get(edge.source)
    const target = nodesById.get(edge.target)
    if (source) source.degree += 1
    if (target) target.degree += 1
  }
  return { nodes: [...nodesById.values()], edges, isTruncated: response.is_truncated ?? false }
}

function edgeWeight(properties: Record<string, unknown>): number {
  const value = Number(properties.weight)
  return Number.isFinite(value) ? value : 1
}

export interface EdgeSizeRange {
  min: number
  max: number
}

const DEFAULT_EDGE_SIZE_RANGE: EdgeSizeRange = { min: 1, max: 5 }

// Absolute log scale, not min-max: weight 1 vs 2 must not render as thinnest vs
// thickest just because those are the only two weights on screen.
export function edgeSize(weight: number, range: EdgeSizeRange): number {
  return Math.min(range.max, range.min + 0.5 * Math.log2(Math.max(weight, 1)))
}

/**
 * Build a brand-new, fully populated graphology instance from raw graph data.
 *
 * IMPORTANT: this always returns a fresh instance. Callers must not mutate a
 * Sigma-bound instance in place (clear + re-add) — Sigma keeps internal
 * render caches (e.g. edge index buffers) that get corrupted by that pattern,
 * causing "edge can't be repaint" errors. Every new dataset must produce a
 * new instance that Sigma binds to once, fully built.
 */
export function sigmaGraphFromRaw(raw: RawGraph, edgeSizeRange: EdgeSizeRange = DEFAULT_EDGE_SIZE_RANGE): {
  graph: MultiUndirectedGraph<NodeDisplayData, EdgeDisplayData>
  typeColors: Map<string, string>
} {
  const graph = new MultiUndirectedGraph<NodeDisplayData, EdgeDisplayData>()
  let typeColors = new Map<string, string>()

  // Degree-based node sizing, matching graph_ui/src/hooks/useLightragGraph.tsx:
  // size = minNodeSize + (maxNodeSize - minNodeSize) * ((degree - minDegree) / range) ** 0.5
  let minDegree = Number.POSITIVE_INFINITY
  let maxDegree = 0
  for (const node of raw.nodes) {
    minDegree = Math.min(minDegree, node.degree)
    maxDegree = Math.max(maxDegree, node.degree)
  }
  const degreeRange = maxDegree - minDegree
  const nodeSizeScale = MAX_NODE_SIZE - MIN_NODE_SIZE

  for (const node of raw.nodes) {
    const initialPosition = hashPosition(node.id)
    const entityType = node.properties.entity_type
    const resolved = resolveNodeColor(typeof entityType === 'string' ? entityType : undefined, typeColors)
    if (resolved.updated) typeColors = resolved.map
    const color = resolved.color
    const size = degreeRange > 0
      ? MIN_NODE_SIZE + nodeSizeScale * Math.pow((node.degree - minDegree) / degreeRange, 0.5)
      : MIN_NODE_SIZE
    graph.addNode(node.id, {
      x: node.x ?? initialPosition.x,
      y: node.y ?? initialPosition.y,
      label: node.labels.join(', ') || node.id,
      color,
      labelColor: color,
      size,
      borderColor: NODE_BORDER_COLOR,
    })
  }

  raw.edges.forEach((edge, index) => {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return
    const key = `edge:${edge.id}:${index}`
    const weight = edgeWeight(edge.properties)
    const size = edgeSize(weight, edgeSizeRange)
    graph.addEdgeWithKey(key, edge.source, edge.target, {
      rawId: edge.id,
      label: typeof edge.properties.keywords === 'string' ? edge.properties.keywords : undefined,
      originalWeight: weight,
      size,
    })
  })
  return { graph, typeColors }
}

export function mergeRawGraphs(current: RawGraph, addition: RawGraph): RawGraph {
  const nodes = new Map(current.nodes.map((node) => [node.id, node]))
  addition.nodes.forEach((node) => nodes.set(node.id, node))
  const edges = new Map(current.edges.map((edge) => [edge.id, edge]))
  addition.edges.forEach((edge) => edges.set(edge.id, edge))
  return graphFromResponse({
    nodes: [...nodes.values()], edges: [...edges.values()],
    is_truncated: current.isTruncated || addition.isTruncated,
  })
}

export function assignCircularLayout(graph: MultiUndirectedGraph<NodeDisplayData, EdgeDisplayData>) {
  const nodeIds = graph.nodes()
  const radius = Math.max(1, nodeIds.length / 12)
  graph.updateEachNodeAttributes((node, attributes) => {
    const index = nodeIds.indexOf(node)
    const angle = (Math.PI * 2 * index) / Math.max(1, nodeIds.length)
    return { ...attributes, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius }
  }, { attributes: ['x', 'y'] })
}

export function assignRandomLayout(graph: MultiUndirectedGraph<NodeDisplayData, EdgeDisplayData>) {
  graph.updateEachNodeAttributes((node, attributes) => ({ ...attributes, ...hashPosition(node) }), { attributes: ['x', 'y'] })
}
