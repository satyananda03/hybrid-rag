import { useKnowledgeGraphStore } from '@/features/knowledge_graph/store'

// English labels ported from graph_ui/src/locales/en.json ("graphPanel.nodeTypes"),
// used to display the same human-readable type names as localhost:8000/webui.
const NODE_TYPE_LABELS: Record<string, string> = {
  person: 'Person',
  category: 'Category',
  geo: 'Geographic',
  location: 'Location',
  organization: 'Organization',
  event: 'Event',
  equipment: 'Equipment',
  weapon: 'Weapon',
  animal: 'Animal',
  unknown: 'Unknown',
  object: 'Object',
  group: 'Group',
  technology: 'Technology',
  product: 'Product',
  document: 'Document',
  content: 'Content',
  data: 'Data',
  artifact: 'Artifact',
  concept: 'Concept',
  naturalobject: 'Natural Object',
  method: 'Method',
  creature: 'Creature',
  plant: 'Plant',
  disease: 'Disease',
  drug: 'Drug',
  food: 'Food',
  table: 'Table',
  drawing: 'Drawing',
  equation: 'Equation',
  other: 'Other',
  db_database: 'Database',
  db_schema: 'Schema',
  db_domain: 'Domain',
  db_table: 'Table',
  db_column: 'Column',
  db_business_concept: 'Business Concept',
  db_value: 'Value',
}

// Each entry also toggles its type's nodes on the canvas.
export function GraphLegend() {
  const colors = useKnowledgeGraphStore((state) => state.typeColors)
  const hidden = useKnowledgeGraphStore((state) => state.settings.hiddenEntityTypes)
  const updateSettings = useKnowledgeGraphStore((state) => state.updateSettings)
  if (colors.size === 0) return null
  const toggle = (type: string) => updateSettings({
    hiddenEntityTypes: hidden.includes(type) ? hidden.filter((t) => t !== type) : [...hidden, type],
  })
  return <aside className="max-h-80 w-48 overflow-auto rounded-lg border border-outline-variant/50 bg-surface-container-high/90 p-2 shadow-lg backdrop-blur-lg">
    <h3 className="mb-2 text-sm font-medium text-on-surface">Legend</h3>
    <div className="flex flex-col gap-1">
      {[...colors].map(([type, color]) => (
        <button
          type="button"
          key={type}
          aria-pressed={!hidden.includes(type)}
          onClick={() => toggle(type)}
          className={`flex items-center gap-2 rounded text-left hover:bg-surface-variant/50 ${hidden.includes(type) ? 'line-through opacity-40' : ''}`}
        >
          <span className="size-4 shrink-0 rounded-full" style={{ backgroundColor: color }} />
          <span className="truncate text-xs" style={{ color }} title={type}>
            {NODE_TYPE_LABELS[type.toLowerCase()] ?? type}
          </span>
        </button>
      ))}
    </div>
  </aside>
}
