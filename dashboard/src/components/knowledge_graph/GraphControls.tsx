import { useState } from 'react'
import {
  BookOpen, Maximize, Minimize, Minus, Plus, RotateCcw, Settings2, Workflow,
} from 'lucide-react'
import { useKnowledgeGraphStore, type GraphLayout } from '@/features/knowledge_graph/store'

interface GraphControlsProps {
  onLayout: (name: GraphLayout) => void
  onZoom: (direction: 'in' | 'out' | 'reset') => void
  onFullscreen: () => void
  isFullscreen: boolean
}

const buttonClass = 'grid size-8 place-items-center rounded text-on-surface-variant hover:bg-surface-variant hover:text-on-surface'
const layouts: Array<{ id: GraphLayout; label: string }> = [
  { id: 'circular', label: 'Circular' },
  { id: 'circlepack', label: 'Circlepack' },
  { id: 'random', label: 'Random' },
  { id: 'forceAtlas2', label: 'ForceAtlas2' },
  { id: 'force', label: 'Force directed' },
  { id: 'noverlap', label: 'Remove overlaps' },
]
const toggles = [
  ['showLabels', 'Show node labels'], ['showEdgeLabels', 'Show edge labels'],
  ['showProperties', 'Show properties'], ['showSearch', 'Show node search'],
  ['enableDrag', 'Drag nodes'], ['hideUnselectedEdges', 'Hide unrelated edges'],
  ['enableEdgeEvents', 'Enable edge selection'],
] as const

export function GraphControls({ onLayout, onZoom, onFullscreen, isFullscreen }: GraphControlsProps) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [layoutsOpen, setLayoutsOpen] = useState(false)
  const settings = useKnowledgeGraphStore((state) => state.settings)
  const edgeCount = useKnowledgeGraphStore((state) => state.edgeCount)
  const updateSettings = useKnowledgeGraphStore((state) => state.updateSettings)

  return <div className="absolute bottom-3 left-3 z-20 flex items-end gap-2">
    <div className="flex flex-col rounded border border-outline-variant/50 bg-surface-container-high/95 p-1 shadow-xl backdrop-blur">
      <div className="relative"><button type="button" className={buttonClass} title="Graph layout" onClick={() => setLayoutsOpen((open) => !open)}><Workflow size={16} /></button>
        {layoutsOpen && <div className="absolute bottom-0 left-9 z-30 w-40 rounded border border-outline-variant/50 bg-surface-container-high p-1 shadow-xl">{layouts.map((layout) => <button key={layout.id} type="button" onClick={() => { onLayout(layout.id); setLayoutsOpen(false) }} className="block w-full rounded px-2 py-1.5 text-left text-xs hover:bg-surface-variant">{layout.label}</button>)}</div>}
      </div>
      <button type="button" className={buttonClass} title="Zoom in" onClick={() => onZoom('in')}><Plus size={16} /></button><button type="button" className={buttonClass} title="Zoom out" onClick={() => onZoom('out')}><Minus size={16} /></button><button type="button" className={buttonClass} title="Reset view" onClick={() => onZoom('reset')}><RotateCcw size={16} /></button>
      <button type="button" className={buttonClass} title="Toggle fullscreen" onClick={onFullscreen}>{isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}</button>
      <button type="button" className={buttonClass} title="Toggle legend" onClick={() => updateSettings({ showLegend: !settings.showLegend })}><BookOpen size={16} /></button>
      <button type="button" className={buttonClass} title="Graph settings" onClick={() => setSettingsOpen((open) => !open)}><Settings2 size={16} /></button>
    </div>
    {settingsOpen && <fieldset className="w-56 space-y-2 rounded border border-outline-variant/50 bg-surface-container-high/95 p-3 text-xs text-on-surface shadow-xl backdrop-blur"><legend className="font-semibold">Graph settings</legend>{toggles.map(([key, label]) => <label key={key} className="flex items-center justify-between gap-3" title={key === 'enableEdgeEvents' && edgeCount > 5000 ? 'Disabled above 5,000 edges for performance' : undefined}><span>{label}</span><input type="checkbox" checked={settings[key]} disabled={key === 'enableEdgeEvents' && edgeCount > 5000} onChange={() => updateSettings({ [key]: !settings[key] })} /></label>)}
      <div className="grid grid-cols-2 gap-2"><NumberSetting label="Min edge size" value={settings.minEdgeSize} max={settings.maxEdgeSize} onChange={(minEdgeSize) => updateSettings({ minEdgeSize })} /><NumberSetting label="Max edge size" value={settings.maxEdgeSize} min={settings.minEdgeSize} max={10} onChange={(maxEdgeSize) => updateSettings({ maxEdgeSize })} /></div>
      <NumberSetting label="Depth" value={settings.maxDepth} onChange={(maxDepth) => updateSettings({ maxDepth })} /><NumberSetting label="Max nodes" value={settings.maxNodes} onChange={(maxNodes) => updateSettings({ maxNodes })} />
    </fieldset>}
  </div>
}

function NumberSetting({ label, value, min = 1, max, onChange }: { label: string; value: number; min?: number; max?: number; onChange: (value: number) => void }) {
  return <label className="block">{label}<input min={min} max={max} type="number" value={value} onChange={(event) => onChange(Math.min(max ?? Number.POSITIVE_INFINITY, Math.max(min, Number(event.target.value) || min)))} className="mt-1 w-full rounded bg-surface-container-low px-1 py-1" /></label>
}
