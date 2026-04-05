import Image from "next/image";

const GITHUB_URL = "https://github.com/calebthecm/MindVault";
const PYPI_URL = "https://pypi.org/project/mindvault/";

const features = [
  {
    icon: "⬇",
    title: "Ingest",
    body: "Claude, ChatGPT, Obsidian vaults, PDFs, plain text. Drop in any folder.",
  },
  {
    icon: "🧠",
    title: "Remember",
    body: "Extracts entities, decisions, and goals from every conversation automatically.",
  },
  {
    icon: "🔍",
    title: "Retrieve",
    body: "Hybrid scoring — summaries first, raw chunks as fallback, entity boosting.",
  },
  {
    icon: "💬",
    title: "Chat",
    body: "Six reasoning modes: CHAT, PLAN, DECIDE, DEBATE, REFLECT, EXPLORE.",
  },
  {
    icon: "🌐",
    title: "Web Search",
    body: "Auto-triggers DuckDuckGo when memory confidence is low. No API key.",
  },
  {
    icon: "🔒",
    title: "Private",
    body: "Everything runs locally. Nothing leaves your machine. Private vault is always separate.",
  },
];

const installMethods = [
  { label: "pip", cmd: "pip install mindvault" },
  { label: "pipx", cmd: "pipx install mindvault" },
  { label: "brew", cmd: "brew install calebthecm/mindvault/mindvault" },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-ink text-white font-body">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-4 border-b border-border max-w-5xl mx-auto">
        <div className="flex items-center gap-3">
          <Image src="/icon-notext.png" alt="MindVault" width={32} height={32} />
          <span className="font-display text-2xl tracking-wider text-white">
            MINDVAULT
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm text-dim">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors"
          >
            GitHub
          </a>
          <a
            href={PYPI_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors"
          >
            PyPI
          </a>
          <a
            href={`${GITHUB_URL}/issues`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors"
          >
            Issues
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-5xl mx-auto px-6 pt-24 pb-20 text-center">
        <div className="flex justify-center mb-8">
          <Image
            src="/icon-text.png"
            alt="MindVault"
            width={220}
            height={220}
            priority
          />
        </div>
        <h1 className="font-display text-6xl md:text-8xl tracking-widest text-white mb-4">
          YOUR BRAIN,{" "}
          <span
            style={{
              background: "linear-gradient(90deg, #ff6a00, #ff8a2a)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            SEARCHABLE
          </span>
        </h1>
        <p className="text-lg text-dim max-w-xl mx-auto mb-10 leading-relaxed">
          A local-first second brain that turns your AI conversations, Obsidian
          notes, and documents into a conversational memory system. Everything
          runs on your machine. No data leaves.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <a
            href={PYPI_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-3 rounded-md text-sm font-medium text-white transition-opacity hover:opacity-90"
            style={{ background: "linear-gradient(90deg, #ff6a00, #ff8a2a)" }}
          >
            Install from PyPI
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-3 rounded-md text-sm font-medium text-white border border-border bg-surface hover:border-blaze transition-colors"
          >
            View on GitHub
          </a>
        </div>
      </section>

      {/* Quick install */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <div className="bg-surface border border-border rounded-xl p-6">
          <p className="text-xs text-dim uppercase tracking-widest mb-4">
            Install
          </p>
          <div className="space-y-3">
            {installMethods.map(({ label, cmd }) => (
              <div key={label} className="flex items-center gap-4">
                <span className="text-xs text-dim w-10 shrink-0">{label}</span>
                <code className="text-sm text-blaze2 font-mono">{cmd}</code>
              </div>
            ))}
          </div>
          <div className="mt-6 pt-5 border-t border-border">
            <p className="text-xs text-dim mb-3">Then:</p>
            <div className="space-y-1">
              <code className="block text-sm text-white font-mono">
                mindvault setup
              </code>
              <code className="block text-sm text-white font-mono">
                mindvault ingest
              </code>
              <code className="block text-sm text-white font-mono">
                mindvault chat
              </code>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-6 pb-24">
        <h2 className="font-display text-4xl tracking-widest text-white mb-10">
          WHAT IT DOES
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map(({ icon, title, body }) => (
            <div
              key={title}
              className="bg-surface border border-border rounded-xl p-5 hover:border-blaze transition-colors"
            >
              <div className="text-2xl mb-3">{icon}</div>
              <h3 className="font-display text-xl tracking-widest text-white mb-2">
                {title.toUpperCase()}
              </h3>
              <p className="text-sm text-dim leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Modes */}
      <section className="max-w-5xl mx-auto px-6 pb-24">
        <h2 className="font-display text-4xl tracking-widest text-white mb-3">
          REASONING MODES
        </h2>
        <p className="text-dim text-sm mb-8">
          Cycle with Shift+Tab in the prompt bar.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            ["💬", "CHAT", "Standard RAG — retrieve and synthesize"],
            ["📋", "PLAN", "Break any task into actionable steps"],
            ["🗳", "DECIDE", "Five-voice council votes on the question"],
            ["⚖", "DEBATE", "FOR vs AGAINST, then a verdict"],
            ["🔍", "REFLECT", "Deep synthesis across your full memory"],
            ["🕸", "EXPLORE", "Graph traversal — follow memory links"],
          ].map(([icon, name, desc]) => (
            <div
              key={name}
              className="bg-surface border border-border rounded-lg p-4"
            >
              <div className="flex items-center gap-2 mb-1">
                <span>{icon}</span>
                <span className="font-display tracking-widest text-sm text-blaze2">
                  {name}
                </span>
              </div>
              <p className="text-xs text-dim">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-5xl mx-auto px-6 pb-24 text-center">
        <div
          className="rounded-2xl p-10 border"
          style={{ borderColor: "#ff6a0040" }}
        >
          <h2 className="font-display text-5xl tracking-widest text-white mb-4">
            OPEN SOURCE
          </h2>
          <p className="text-dim text-sm mb-8 max-w-md mx-auto">
            MindVault is source-available and actively developed. File issues,
            request features, or contribute.
          </p>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-md text-sm font-medium text-white border border-border bg-surface hover:border-blaze transition-colors"
          >
            calebthecm/MindVault ↗
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6 text-center text-xs text-dim max-w-5xl mx-auto">
        <p>
          Built by{" "}
          <a
            href="https://calebmedia.co"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors"
          >
            Caleb Media
          </a>{" "}
          · Local-first · No tracking · No cloud
        </p>
      </footer>
    </main>
  );
}
