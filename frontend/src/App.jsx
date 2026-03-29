import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import { Analytics } from '@vercel/analytics/react';

const API = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

function getSessionId() {
  let sid = localStorage.getItem('finsight_session_id');
  if (!sid) {
    sid = uuidv4();
    localStorage.setItem('finsight_session_id', sid);
  }
  return sid;
}

const SESSION_ID = getSessionId();

const SUGGESTION_CHIPS = [
  'What were the main revenue drivers?',
  'What risks could impact earnings?',
  'How did operating margins change YoY?',
  "What is management's growth outlook?",
  'Describe the competitive landscape',
  'What are the key R&D investments?',
];

const PROGRESS_STEPS = [
  'Preparing analysis...',
  'Loading filing...',
  'Processing document...',
  'Generating summary (1/5)',
  'Generating summary (2/5)',
  'Generating summary (3/5)',
  'Generating summary (4/5)',
  'Generating summary (5/5)',
  'Generating financial charts...',
  'Complete',
];

const PROGRESS_MAP = [
  ['Fetching filing',    'Loading filing from SEC...'],
  ['Extracting text',   'Reading filing content...'],
  ['Building vector',   'Processing & indexing document...'],
  ['Generating financial', 'Building financial charts...'],
];

function friendlyProgress(raw) {
  if (!raw) return raw;
  for (const [key, label] of PROGRESS_MAP) {
    if (raw.includes(key)) return label;
  }
  if (raw.includes('Generating summary')) return raw.replace('Generating summary', 'Summarizing section');
  return raw;
}

function ProgressBar({ progress }) {
  const stepIndex = PROGRESS_STEPS.findIndex(s => progress.includes(s.split('...')[0].split('(')[0].trim()));
  const pct = stepIndex >= 0 ? Math.round(((stepIndex + 1) / PROGRESS_STEPS.length) * 100) : 15;

  return (
    <div className="mb-6">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{friendlyProgress(progress)}</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
        <div
          className="h-1.5 rounded-full transition-all duration-700 progress-shimmer"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function MetricBadge({ label, value, color = 'gray' }) {
  const colors = {
    blue: 'bg-blue-900/30 text-blue-300 border-blue-700/60',
    green: 'bg-green-900/30 text-green-300 border-green-700/60',
    red: 'bg-red-900/30 text-red-300 border-red-700/60',
    gray: 'bg-gray-800/80 text-gray-300 border-gray-700/60',
  };
  const topBar = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    gray: 'bg-gray-600',
  };
  return (
    <div className={`rounded border text-xs overflow-hidden ${colors[color]}`}>
      <div className={`h-0.5 w-full ${topBar[color]}`} />
      <div className="px-3 py-2">
        <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-0.5">{label}</div>
        <div className="font-mono font-semibold">{value || '—'}</div>
      </div>
    </div>
  );
}

function formatFiscalYearEnd(val) {
  if (!val) return '—';
  const s = String(val).replace(/\D/g, '');
  if (s.length === 4) {
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const m = parseInt(s.slice(0, 2), 10);
    const d = s.slice(2);
    if (m >= 1 && m <= 12) return `${months[m - 1]} ${d}`;
  }
  return val;
}

function SectionAccordion({ section, index, open, onToggle }) {
  const icons = ['🏢', '📊', '⚠️', '🚀', '🌐'];

  return (
    <div className={`rounded-lg overflow-hidden transition-all ${open ? 'border border-blue-800/50 shadow-lg shadow-blue-950/30' : 'border border-gray-800'}`}>
      <button
        className={`w-full flex items-center justify-between px-4 py-3 transition-colors text-left ${open ? 'bg-gray-900/80' : 'bg-gray-900 hover:bg-gray-800/80'}`}
        onClick={onToggle}
      >
        <div className="flex items-center gap-2.5">
          {open && <div className="w-0.5 h-5 bg-blue-500 rounded-full" />}
          <span className="text-base">{icons[index] || '📄'}</span>
          <span className={`text-sm font-semibold ${open ? 'text-blue-100' : 'text-gray-100'}`}>{section.title}</span>
        </div>
        <span className={`text-xs transition-transform ${open ? 'text-blue-400 rotate-180' : 'text-gray-500'}`}>▼</span>
      </button>
      {open && (
        <div className="px-4 py-3 bg-gray-950 border-t border-blue-900/30">
          <div className="prose prose-invert prose-base max-w-none text-gray-200 leading-relaxed">
            <ReactMarkdown>{section.content}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryAccordionGroup({ summary, formType, onExportPdf }) {
  const [openSections, setOpenSections] = useState(summary.map((_, i) => i === 0));
  const allExpanded = openSections.every(Boolean);
  const allCollapsed = openSections.every(v => !v);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-5 bg-blue-500 rounded" />
        <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">
          {formType} Summary
        </h3>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setOpenSections(openSections.map(() => true))}
            disabled={allExpanded}
            className="text-xs text-gray-400 hover:text-blue-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Expand All
          </button>
          <span className="text-gray-700">|</span>
          <button
            onClick={() => setOpenSections(openSections.map(() => false))}
            disabled={allCollapsed}
            className="text-xs text-gray-400 hover:text-blue-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Collapse All
          </button>
          <button
            onClick={onExportPdf}
            className="text-xs bg-gray-800 text-gray-400 border border-gray-700 px-3 py-1.5 rounded hover:bg-gray-700 hover:text-gray-200 hover:border-gray-600 transition-colors flex items-center gap-1"
            title="Download PDF report"
          >
            📥 Export PDF
          </button>
        </div>
      </div>
      <div className="space-y-2">
        {summary.map((section, i) => (
          <SectionAccordion
            key={i}
            section={section}
            index={i}
            open={openSections[i]}
            onToggle={() => setOpenSections(s => s.map((v, j) => j === i ? !v : v))}
          />
        ))}
      </div>
    </div>
  );
}

function ChatBubble({ role, content, cited, source, filingType = '10-K' }) {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`max-w-[85%] px-4 py-3 rounded-xl text-sm ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-none'
            : 'bg-gray-800 text-gray-200 rounded-bl-none border border-gray-700'
        }`}
      >
        {isUser ? (
          <p>{content}</p>
        ) : (
          <>
            <div className="prose prose-invert prose-base max-w-none leading-relaxed text-gray-100">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
            {cited && cited.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-700">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider">Sources: </span>
                {cited.map((s, i) => (
                  <span
                    key={i}
                    className="inline-block text-[10px] bg-gray-700 text-blue-300 px-2 py-0.5 rounded mr-1 mt-1"
                  >
                    {s}
                  </span>
                ))}
              </div>
            )}
            {source && (
              <div className="mt-1">
                <span className={`text-[10px] ${source === 'realtime' ? 'text-green-400' : source === 'hybrid' ? 'text-yellow-400' : 'text-gray-500'}`}>
                  {source === 'realtime' ? '⚡ Live Data' : source === 'hybrid' ? '⚡ Live + Filing' : `📄 ${filingType} Filing`}
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function CompareAccordion({ section1, section2, ticker1, ticker2, index, open, onToggle }) {
  const icons = ['🏢', '📊', '⚠️', '🚀', '🌐'];
  return (
    <div className={`rounded-lg overflow-hidden transition-all ${open ? 'border border-green-800/40 shadow-lg shadow-green-950/20' : 'border border-gray-800'}`}>
      <button
        className={`w-full flex items-center justify-between px-4 py-3 transition-colors text-left ${open ? 'bg-gray-900/80' : 'bg-gray-900 hover:bg-gray-800/80'}`}
        onClick={onToggle}
      >
        <div className="flex items-center gap-2.5">
          {open && <div className="w-0.5 h-5 bg-green-500 rounded-full" />}
          <span className="text-base">{icons[index] || '📄'}</span>
          <span className={`text-sm font-semibold ${open ? 'text-green-100' : 'text-gray-100'}`}>{section1.title}</span>
        </div>
        <span className={`text-xs transition-transform ${open ? 'text-green-400 rotate-180' : 'text-gray-500'}`}>▼</span>
      </button>
      {open && (
        <div className="grid grid-cols-2 bg-gray-950 border-t border-green-900/20 divide-x divide-gray-800">
          <div className="px-4 py-3">
            <div className="text-[10px] text-blue-400 font-mono font-bold mb-2 uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block" />{ticker1}
            </div>
            <div className="prose prose-invert prose-base max-w-none text-gray-200 leading-relaxed">
              <ReactMarkdown>{section1.content}</ReactMarkdown>
            </div>
          </div>
          <div className="px-4 py-3">
            <div className="text-[10px] text-green-400 font-mono font-bold mb-2 uppercase tracking-wider flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />{ticker2}
            </div>
            <div className="prose prose-invert prose-base max-w-none text-gray-200 leading-relaxed">
              <ReactMarkdown>{section2.content}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CompareAccordionGroup({ comparisonData, onExportPdf }) {
  const count = comparisonData.company1.summary.length;
  const [openSections, setOpenSections] = useState(Array.from({ length: count }, (_, i) => i === 0));
  const allExpanded = openSections.every(Boolean);
  const allCollapsed = openSections.every(v => !v);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-5 bg-green-500 rounded" />
        <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">
          Side-by-Side Comparison
        </h3>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setOpenSections(openSections.map(() => true))}
            disabled={allExpanded}
            className="text-xs text-gray-400 hover:text-green-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Expand All
          </button>
          <span className="text-gray-700">|</span>
          <button
            onClick={() => setOpenSections(openSections.map(() => false))}
            disabled={allCollapsed}
            className="text-xs text-gray-400 hover:text-green-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Collapse All
          </button>
          <button
            onClick={onExportPdf}
            className="text-xs bg-gray-800 text-gray-400 border border-gray-700 px-3 py-1.5 rounded hover:bg-gray-700 hover:text-gray-200 hover:border-gray-600 transition-colors flex items-center gap-1"
            title="Download comparison PDF"
          >
            📥 Export PDF
          </button>
        </div>
      </div>
      <div className="space-y-2">
        {comparisonData.company1.summary.map((s1, i) => {
          const s2 = comparisonData.company2.summary[i] || { title: s1.title, content: 'Not available.' };
          return (
            <CompareAccordion
              key={i}
              section1={s1}
              section2={s2}
              ticker1={comparisonData.ticker1}
              ticker2={comparisonData.ticker2}
              index={i}
              open={openSections[i]}
              onToggle={() => setOpenSections(s => s.map((v, j) => j === i ? !v : v))}
            />
          );
        })}
      </div>
    </div>
  );
}

function HelpModal({ onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-bold text-gray-100">How to use FinSight AI</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-200 text-xl leading-none">✕</button>
        </div>

        <div className="space-y-4 text-sm text-gray-300">
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">1.</span>
            <div><span className="text-gray-100 font-semibold">Search a company</span> - type any company name or ticker (e.g. Apple, TSLA). Select from the dropdown.</div>
          </div>
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">2.</span>
            <div><span className="text-gray-100 font-semibold">Choose filing & mode</span> - toggle <span className="text-blue-300">10-K / 10-Q</span> and <span className="text-blue-300">Executive / Analyst</span> before selecting a company.</div>
          </div>
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">3.</span>
            <div><span className="text-gray-100 font-semibold">Read the 5-section summary</span> - Business Overview, Financial Performance, Risk Factors, Strategic Initiatives, Market Position. Expand / collapse sections individually or all at once.</div>
          </div>
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">4.</span>
            <div><span className="text-gray-100 font-semibold">Ask follow-up questions</span> - the Q&A panel streams answers from the filing. Conversation history is retained for the session. Click a suggestion chip to get started.</div>
          </div>
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">5.</span>
            <div><span className="text-gray-100 font-semibold">Compare two companies</span> - click <span className="text-green-400">⚖️ Compare</span> and search a second company for a side-by-side breakdown.</div>
          </div>
          <div className="flex gap-3">
            <span className="text-blue-400 font-bold w-5 shrink-0">6.</span>
            <div><span className="text-gray-100 font-semibold">Explore charts & export</span> - view 5-year financial charts, live news sentiment, and export a full PDF report.</div>
          </div>
        </div>

        <div className="mt-5 pt-4 border-t border-gray-800 text-[11px] text-gray-600">
          Data sourced from SEC EDGAR · Not financial advice
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [analyzeMode, setAnalyzeMode] = useState('analyst');
  const [activeChart, setActiveChart] = useState('income_statement');

  // Q&A state
  const [chatHistory, setChatHistory] = useState([]);
  const [qaInput, setQaInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [providerName, setProviderName] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const [cachedAnalyses, setCachedAnalyses] = useState([]);   // [{ticker, formType, mode}]
  const [indexOnlyTickers, setIndexOnlyTickers] = useState([]); // [{ticker, formType}]
  const [homeRefreshKey, setHomeRefreshKey] = useState(0);
  const [formType, setFormType] = useState('10-K');

  // Q&A company selector (in compare mode)
  const [qaCompany, setQaCompany] = useState('primary'); // 'primary' | 'compare'

  // Compare mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareQuery, setCompareQuery] = useState('');
  const [compareResults, setCompareResults] = useState([]);
  const [compareCompany, setCompareCompany] = useState(null);
  const [compareJobId, setCompareJobId] = useState(null);
  const [compareJobStatus, setCompareJobStatus] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  // News sentiment
  const [newsSentiment, setNewsSentiment] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);

  const pollRef = useRef(null);
  const comparePollRef = useRef(null);
  const chatEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const hasMountedRef = useRef(false);

  // Derived from job result - declared before useEffect hooks that reference them
  const result = jobStatus?.result;
  const profile = result?.company_profile;
  const summary = result?.summary;
  const charts = result?.financial_charts;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, streamingContent]);

  useEffect(() => {
    axios.get(`${API}/health`).then(r => setProviderName(r.data.provider)).catch(() => {});
  }, []);

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/cache/`).catch(() => ({ data: { cached: {} } })),
      axios.get(`${API}/vector_stores/`).catch(() => ({ data: { available_stores: [] } })),
    ]).then(([cacheRes, storeRes]) => {
      const cached = cacheRes.data.cached || {};
      const analyses = [];
      const cachedCombos = new Set();
      for (const [ticker, variants] of Object.entries(cached)) {
        for (const variant of variants) {
          const m = variant.match(/^([\w-]+)\s+\((\w+)\)$/);
          if (m) {
            const ft = m[1], mode = m[2].toLowerCase();
            analyses.push({ ticker, formType: ft, mode });
            cachedCombos.add(`${ticker}:${ft}`);
          }
        }
      }
      setCachedAnalyses(analyses.sort((a, b) => a.ticker.localeCompare(b.ticker)));
      const stores = storeRes.data.available_stores || [];
      const indexOnly = stores
        .filter(s => !cachedCombos.has(s))
        .map(s => { const [ticker, formType] = s.split(':'); return { ticker, formType }; })
        .sort((a, b) => a.ticker.localeCompare(b.ticker));
      setIndexOnlyTickers(indexOnly);
    });
  }, [homeRefreshKey]);

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/job_status/${jobId}`);
        setJobStatus(res.data);
        if (res.data.status === 'complete' || res.data.status === 'error') {
          clearInterval(pollRef.current);
        }
      } catch (e) {
        clearInterval(pollRef.current);
      }
    }, 2000);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  useEffect(() => {
    if (!compareJobId) return;
    comparePollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/job_status/${compareJobId}`);
        setCompareJobStatus(res.data);
        if (res.data.status === 'complete' || res.data.status === 'error') {
          clearInterval(comparePollRef.current);
        }
      } catch {
        clearInterval(comparePollRef.current);
      }
    }, 2000);
    return () => clearInterval(comparePollRef.current);
  }, [compareJobId]);

  useEffect(() => {
    if (
      jobStatus?.status === 'complete' &&
      compareJobStatus?.status === 'complete' &&
      profile &&
      compareCompany
    ) {
      setComparisonLoading(true);
      axios.get(`${API}/compare/?ticker1=${profile.ticker}&ticker2=${compareCompany.ticker}`)
        .then(r => setComparisonData(r.data))
        .catch(e => {
          const msg = e.response?.data?.detail;
          if (msg) alert(msg);
          setComparisonData(null);
        })
        .finally(() => setComparisonLoading(false));
    }
  }, [jobStatus?.status, compareJobStatus?.status]); // eslint-disable-line

  useEffect(() => {
    if (jobStatus?.status === 'complete' && profile?.ticker) {
      setNewsLoading(true);
      setNewsSentiment(null);
      axios.get(`${API}/news_sentiment/?ticker=${profile.ticker}`)
        .then(r => setNewsSentiment(r.data))
        .catch(() => setNewsSentiment(null))
        .finally(() => setNewsLoading(false));
    }
  }, [jobStatus?.status, profile?.ticker]); // eslint-disable-line

  // Re-run analysis when mode or filing type changes (if a company is already selected)
  useEffect(() => {
    if (!hasMountedRef.current) { hasMountedRef.current = true; return; }
    if (!selectedCompany) return;
    setJobStatus({ status: 'processing', progress: 'Preparing analysis...' });
    setJobId(null);
    setChatHistory([]);
    setStreamingContent('');
    setCompareMode(false);
    setCompareCompany(null);
    setCompareJobId(null);
    setCompareJobStatus(null);
    setComparisonData(null);
    setNewsSentiment(null);
    axios.post(`${API}/analyze_filing/?ticker=${selectedCompany.ticker}&analyze_mode=${analyzeMode}&form_type=${formType}`)
      .then(res => { setJobId(res.data.job_id); setJobStatus({ status: 'processing', progress: 'Starting...' }); })
      .catch(() => setJobStatus({ status: 'error', progress: 'Failed to start analysis' }));
  }, [analyzeMode, formType]); // eslint-disable-line

  const searchAbortRef = useRef(null);
  const compareSearchAbortRef = useRef(null);
  const handleSearch = useCallback(async (q) => {
    if (!q || q.length < 2) { setSearchResults([]); return; }
    if (searchAbortRef.current) searchAbortRef.current.abort();
    searchAbortRef.current = new AbortController();
    try {
      const res = await axios.get(`${API}/company_search/?q=${encodeURIComponent(q)}`, {
        signal: searchAbortRef.current.signal,
      });
      // Deduplicate by CIK to prevent race-condition phantom duplicates
      const seen = new Set();
      const deduped = (res.data.results || []).filter(c => {
        if (seen.has(c.cik)) return false;
        seen.add(c.cik);
        return true;
      });
      setSearchResults(deduped);
    } catch (e) {
      if (!axios.isCancel(e)) setSearchResults([]);
    }
  }, []);

  const handleCompanySelect = useCallback(async (company) => {
    setSelectedCompany(company);
    setQuery('');
    setSearchResults([]);
    setJobId(null);
    setChatHistory([]);
    setStreamingContent('');
    setCompareMode(false);
    setCompareCompany(null);
    setCompareJobId(null);
    setCompareJobStatus(null);
    setComparisonData(null);
    setNewsSentiment(null);
    setQaCompany('primary');
    setJobStatus({ status: 'processing', progress: 'Preparing analysis...' });

    try {
      const res = await axios.post(
        `${API}/analyze_filing/?ticker=${company.ticker}&analyze_mode=${analyzeMode}&form_type=${formType}`
      );
      setJobId(res.data.job_id);
      setJobStatus({ status: 'processing', progress: 'Starting...' });
    } catch {
      setJobStatus({ status: 'error', progress: 'Failed to start analysis' });
    }
  }, [analyzeMode, formType]);

  const handleQuickLoad = useCallback(async (ticker, ft, mode) => {
    setSelectedCompany({ ticker, title: ticker });
    setFormType(ft);
    setAnalyzeMode(mode);
    setQuery('');
    setSearchResults([]);
    setJobId(null);
    setChatHistory([]);
    setStreamingContent('');
    setCompareMode(false);
    setCompareCompany(null);
    setCompareJobId(null);
    setCompareJobStatus(null);
    setComparisonData(null);
    setNewsSentiment(null);
    setQaCompany('primary');
    setJobStatus({ status: 'processing', progress: 'Preparing analysis...' });
    try {
      const res = await axios.post(`${API}/analyze_filing/?ticker=${ticker}&analyze_mode=${mode}&form_type=${ft}`);
      setJobId(res.data.job_id);
      setJobStatus({ status: 'processing', progress: 'Starting...' });
    } catch {
      setJobStatus({ status: 'error', progress: 'Failed to start analysis' });
    }
  }, []);

  const handleCompareSearch = useCallback(async (q) => {
    if (!q || q.length < 2) { setCompareResults([]); return; }
    if (compareSearchAbortRef.current) compareSearchAbortRef.current.abort();
    compareSearchAbortRef.current = new AbortController();
    try {
      const res = await axios.get(`${API}/company_search/?q=${encodeURIComponent(q)}`, {
        signal: compareSearchAbortRef.current.signal,
      });
      const seen = new Set();
      const deduped = (res.data.results || []).filter(c => {
        if (seen.has(c.cik)) return false;
        seen.add(c.cik);
        return true;
      });
      setCompareResults(deduped);
    } catch (e) {
      if (!axios.isCancel(e)) setCompareResults([]);
    }
  }, []);

  const handleCompareSelect = useCallback(async (company) => {
    if (company.ticker === selectedCompany?.ticker) {
      alert(`${company.ticker} is already the primary company. Please choose a different company to compare.`);
      return;
    }
    setCompareCompany(company);
    setCompareQuery('');
    setCompareResults([]);
    setCompareJobId(null);
    setComparisonData(null);
    setCompareJobStatus({ status: 'processing', progress: 'Preparing analysis...' });

    try {
      const res = await axios.post(
        `${API}/analyze_filing/?ticker=${company.ticker}&analyze_mode=${analyzeMode}&form_type=${formType}`
      );
      setCompareJobId(res.data.job_id);
      setCompareJobStatus({ status: 'processing', progress: 'Starting...' });
    } catch {
      setCompareJobStatus({ status: 'error', progress: 'Failed to start analysis' });
    }
  }, [analyzeMode, formType, selectedCompany?.ticker]);

  const handleAsk = useCallback((question) => {
    if (!question.trim() || !selectedCompany || isStreaming) return;

    const q = question.trim();
    setQaInput('');
    setChatHistory(h => [...h, { role: 'user', content: q }]);
    setIsStreaming(true);
    setStreamingContent('');

    if (eventSourceRef.current) eventSourceRef.current.close();

    const askTicker = (compareCompany && qaCompany === 'compare') ? compareCompany.ticker : selectedCompany.ticker;
    const askFormType = (compareCompany && qaCompany === 'compare')
      ? (compareJobStatus?.result?.form_type || formType)
      : (result?.form_type || formType);
    const url = `${API}/ask/?question=${encodeURIComponent(q)}&ticker=${askTicker}&session_id=${SESSION_ID}&form_type=${encodeURIComponent(askFormType)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    let accumulated = '';
    let citedSections = [];
    let source = 'document';

    es.onmessage = (event) => {
      if (event.data === '[DONE]') {
        es.close();
        return;
      }
      try {
        const data = JSON.parse(event.data);
        if (data.token) {
          accumulated += data.token;
          setStreamingContent(accumulated);
        }
        if (data.done) {
          citedSections = data.cited_sections || [];
          source = data.source || 'document';
          setChatHistory(h => [
            ...h,
            { role: 'assistant', content: accumulated, cited: citedSections, source },
          ]);
          setStreamingContent('');
          setIsStreaming(false);
          es.close();
        }
        if (data.error) {
          setChatHistory(h => [
            ...h,
            { role: 'assistant', content: `Error: ${data.error}`, cited: [], source: 'document' },
          ]);
          setStreamingContent('');
          setIsStreaming(false);
          es.close();
        }
      } catch {}
    };

    es.onerror = () => {
      if (accumulated) {
        setChatHistory(h => [
          ...h,
          { role: 'assistant', content: accumulated, cited: [], source: 'document' },
        ]);
      }
      setStreamingContent('');
      setIsStreaming(false);
      es.close();
    };
  }, [selectedCompany, isStreaming, compareCompany, qaCompany, result, compareJobStatus, formType]);

  const handleExportPdf = useCallback(async () => {
    if (!profile) return;
    try {
      const payload = { ...result, news_sentiment: newsSentiment };
      const res = await fetch(`${API}/export_pdf/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Export failed');
      const blob = await res.blob();
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename=([^;]+)/);
      const filename = match ? match[1] : `FinSight_AI_${profile.ticker}.pdf`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      alert('PDF export failed. Please try again.');
    }
  }, [profile, result, analyzeMode, formType, newsSentiment]);

  const handleExportComparePdf = useCallback(async () => {
    if (!comparisonData) return;
    try {
      const res = await fetch(`${API}/export_pdf/compare/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...comparisonData, analyze_mode: analyzeMode }),
      });
      if (!res.ok) throw new Error('Export failed');
      const blob = await res.blob();
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename=([^;]+)/);
      const filename = match ? match[1] : `FinSight_AI_Compare_${comparisonData.ticker1}_vs_${comparisonData.ticker2}.pdf`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      alert('Comparison PDF export failed. Please try again.');
    }
  }, [comparisonData]);

  return (
    <>
    <div className="min-h-screen bg-gray-950 text-gray-100 font-mono flex flex-col">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="border-b border-gray-800/80 bg-gray-950/90 backdrop-blur-sm px-6 py-3 flex items-center justify-between sticky top-0 z-50">
        <button
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          onClick={() => {
            setSelectedCompany(null);
            setJobId(null);
            setJobStatus(null);
            setChatHistory([]);
            setStreamingContent('');
            setCompareMode(false);
            setCompareCompany(null);
            setCompareJobId(null);
            setCompareJobStatus(null);
            setComparisonData(null);
            setNewsSentiment(null);
            setQuery('');
            setSearchResults([]);
            setHomeRefreshKey(k => k + 1);
          }}
          title="Back to Home"
        >
          <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            FinSight AI
          </span>
          <span className="hidden sm:inline text-gray-700 text-xs">|</span>
          <span className="hidden sm:inline text-gray-500 text-[11px] tracking-wide">SEC Filing Analysis · 10-K / 10-Q</span>
        </button>
        <div className="flex items-center gap-3">
          {providerName && (
            <span className="text-[11px] bg-blue-950/60 text-blue-300 border border-blue-800/60 px-2.5 py-1 rounded-full">
              ⚡ {providerName}
            </span>
          )}
          <button
            onClick={() => setShowHelp(true)}
            className="text-gray-500 hover:text-gray-200 text-sm border border-gray-700 rounded-full w-6 h-6 flex items-center justify-center hover:border-gray-500 transition-colors"
            title="How to use"
          >?</button>
        </div>
      </header>
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-6 space-y-6">
        {/* ── Search ───────────────────────────────────────────────────────── */}
        <div className="relative">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <label className="text-xs text-gray-400 uppercase tracking-wider">Company Search</label>
            <div className="flex gap-1 ml-auto flex-wrap">
              {/* Filing type */}
              <div className="flex gap-1 border border-gray-700 rounded overflow-hidden mr-2">
                {['10-K', '10-Q'].map(ft => (
                  <button
                    key={ft}
                    onClick={() => setFormType(ft)}
                    className={`px-3 py-1 text-xs transition-colors ${
                      formType === ft ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    {ft}
                  </button>
                ))}
              </div>
              {/* Analysis depth */}
              <div className="flex gap-1 border border-gray-700 rounded overflow-hidden">
                {[['executive', 'Executive'], ['analyst', 'Analyst']].map(([mode, label]) => (
                  <button
                    key={mode}
                    onClick={() => setAnalyzeMode(mode)}
                    className={`px-3 py-1 text-xs transition-colors ${
                      analyzeMode === mode ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
              <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <circle cx="11" cy="11" r="7" />
                <line x1="16.5" y1="16.5" x2="21" y2="21" strokeLinecap="round" />
              </svg>
            </span>
            <input
              type="text"
              placeholder="Enter company name or ticker (e.g., Tesla, TSLA)"
              value={query}
              onChange={e => { setQuery(e.target.value); handleSearch(e.target.value); }}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition-colors"
            />
          </div>
          {query && searchResults.length > 0 && (
            <ul className="absolute z-10 w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg max-h-60 overflow-y-auto shadow-2xl">
              {searchResults.map(c => (
                <li
                  key={c.cik}
                  onClick={() => handleCompanySelect(c)}
                  className="px-4 py-3 cursor-pointer hover:bg-gray-800 border-b border-gray-800 last:border-b-0 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-100">{c.title}</span>
                    <span className="text-xs bg-blue-900/40 text-blue-300 border border-blue-800 px-2 py-0.5 rounded font-mono">
                      {c.ticker}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* ── Empty state ──────────────────────────────────────────────────── */}
        {!selectedCompany && (
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-4">How it works</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5 items-stretch">
              {[
                { step: '1', title: 'Search a company', desc: 'Type any name or ticker - Apple, TSLA, JPMorgan' },
                { step: '2', title: 'Get a structured summary', desc: '5-section analysis: Business, Financials, Risks, Strategy, Market Position' },
                { step: '3', title: 'Ask follow-up questions', desc: 'Multi-turn Q&A streamed directly from the filing' },
              ].map(({ step, title, desc }) => (
                <div key={step} className="flex gap-3 bg-gray-800/50 border border-gray-700/50 rounded-lg p-3 h-full">
                  <span className="text-blue-400 font-bold text-sm shrink-0">{step}.</span>
                  <div>
                    <p className="text-sm text-gray-200 font-medium">{title}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-3 gap-x-4 gap-y-2 pt-4 border-t border-gray-800">
              {['10-K & 10-Q', 'Compare 2 Companies', '5-Year Charts', 'News Sentiment', 'PDF Export', 'Executive / Analyst Mode'].map(f => (
                <span key={f} className="flex items-center gap-1.5 text-[11px] text-gray-500"><span className="w-1 h-1 rounded-full bg-blue-500/60 shrink-0"></span>{f}</span>
              ))}
            </div>
            {cachedAnalyses.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-800">
                <p className="text-[11px] text-gray-500 uppercase tracking-wider mb-3">
                  Summary Ready <span className="normal-case text-green-500/70 ml-1">- Loads Instantly</span>
                </p>
                {['10-K', '10-Q'].map(ft => {
                  const analyst = cachedAnalyses.filter(a => a.formType === ft && a.mode === 'analyst');
                  const executive = cachedAnalyses.filter(a => a.formType === ft && a.mode === 'executive');
                  if (!analyst.length && !executive.length) return null;
                  return (
                    <div key={ft} className="mb-3">
                      <p className="text-[11px] text-gray-600 font-mono mb-1.5">{ft}</p>
                      {executive.length > 0 && (
                        <div className="flex items-center gap-2 mb-1.5 ml-3">
                          <span className="text-[10px] text-gray-600 w-16 shrink-0">Executive</span>
                          <div className="flex flex-wrap gap-1.5">
                            {executive.map(({ ticker }) => (
                              <button key={`${ticker}:${ft}:executive`} onClick={() => handleQuickLoad(ticker, ft, 'executive')}
                                className="text-[11px] font-mono bg-green-500/10 text-green-400 border border-green-500/20 px-2.5 py-1 rounded hover:bg-green-500/20 transition-colors">
                                {ticker}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {analyst.length > 0 && (
                        <div className="flex items-center gap-2 ml-3">
                          <span className="text-[10px] text-gray-600 w-16 shrink-0">Analyst</span>
                          <div className="flex flex-wrap gap-1.5">
                            {analyst.map(({ ticker }) => (
                              <button key={`${ticker}:${ft}:analyst`} onClick={() => handleQuickLoad(ticker, ft, 'analyst')}
                                className="text-[11px] font-mono bg-green-500/10 text-green-400 border border-green-500/20 px-2.5 py-1 rounded hover:bg-green-500/20 transition-colors">
                                {ticker}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            {indexOnlyTickers.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-800">
                <p className="text-[11px] text-gray-500 uppercase tracking-wider mb-3">
                  Index Built <span className="normal-case text-violet-400/70 ml-1">- Ready to Summarize (~30 secs)</span>
                </p>
                {['10-K', '10-Q'].map(ft => {
                  const entries = indexOnlyTickers.filter(t => t.formType === ft);
                  if (!entries.length) return null;
                  return (
                    <div key={ft} className="mb-3">
                      <p className="text-[11px] text-gray-600 font-mono mb-1.5">{ft}</p>
                      <div className="flex flex-wrap gap-1.5 ml-3">
                        {entries.map(({ ticker }) => (
                          <button
                            key={`${ticker}:${ft}`}
                            onClick={() => handleQuickLoad(ticker, ft, analyzeMode)}
                            className="text-[11px] font-mono bg-violet-500/10 text-violet-400 border border-violet-500/20 px-2.5 py-1 rounded hover:bg-violet-500/20 transition-colors"
                          >
                            {ticker}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            {cachedAnalyses.length === 0 && indexOnlyTickers.length === 0 && (
              <p className="text-[11px] text-gray-600 mt-3">Previously analyzed companies load instantly. New companies take 1-2 minutes to fetch the filing and build the index.</p>
            )}
            <p className="text-[11px] text-gray-700 mt-3 pt-3 border-t border-gray-800/60">
              The lists above are not exhaustive - they show companies with pre-built indexes for faster loading. Any public company on SEC can be searched using the search bar above. For companies not listed above, the vector index will be built on first analysis.
            </p>
          </div>
        )}

        {/* ── Selected company loading card ────────────────────────────────── */}
        {selectedCompany && jobStatus?.status === 'processing' && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-4 flex items-center gap-3">
            <div className="flex gap-1">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
            <div>
              <span className="text-sm font-semibold text-gray-100">{selectedCompany.title}</span>
              <span className="text-xs text-gray-500 ml-2 font-mono">{selectedCompany.ticker}</span>
              <div className="text-xs text-gray-500 mt-0.5">
                {formType} · {analyzeMode === 'analyst' ? 'Analyst' : 'Executive'}
              </div>
            </div>
          </div>
        )}

        {/* ── Progress bar ─────────────────────────────────────────────────── */}
        {jobStatus?.status === 'processing' && (
          <ProgressBar progress={jobStatus.progress} />
        )}

        {/* ── Error ────────────────────────────────────────────────────────── */}
        {jobStatus?.status === 'error' && (
          <div className="bg-red-900/20 border border-red-800 rounded-lg px-4 py-3 text-red-300 text-sm">
            ✗ {jobStatus.error || jobStatus.progress}
          </div>
        )}

        {/* ── Company card ─────────────────────────────────────────────────── */}
        {profile && jobStatus?.status === 'complete' && (
          <div className="bg-gray-900 border border-gray-800 border-l-2 border-l-blue-500 rounded-xl p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-50 mb-1">{profile.name}</h2>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs bg-blue-900/40 text-blue-300 border border-blue-800 px-2 py-0.5 rounded font-mono">
                    {profile.ticker}
                  </span>
                  {profile.exchanges?.map(ex => (
                    <span key={ex} className="text-xs text-gray-500">{ex}</span>
                  ))}
                </div>
              </div>
              <div className="flex flex-col items-end gap-2">
                <div className="text-right">
                  <div className="text-[10px] text-gray-600 uppercase tracking-wider">Analysis Mode</div>
                  <div className="text-xs text-blue-300 font-semibold">
                    {result?.form_type || formType} · {(result?.analyze_mode || analyzeMode) === 'analyst' ? 'Analyst' : 'Executive'}
                  </div>
                </div>
                <button
                  onClick={() => {
                    const entering = !compareMode;
                    setCompareMode(entering);
                    setCompareQuery('');
                    setCompareResults([]);
                    setCompareCompany(null);
                    setCompareJobId(null);
                    setCompareJobStatus(null);
                    setComparisonData(null);
                    // Cancelling compare: reset Q&A to primary company
                    if (!entering) {
                      setQaCompany('primary');
                      setChatHistory([]);
                      setStreamingContent('');
                    }
                  }}
                  className={`text-xs px-3 py-1.5 rounded border transition-colors ${
                    compareMode
                      ? 'bg-green-900/40 text-green-300 border-green-700'
                      : 'bg-gray-800 text-gray-400 border-gray-700 hover:text-gray-200'
                  }`}
                >
                  {compareMode ? '✖ Cancel Compare' : '⚖️ Compare'}
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <MetricBadge label="CIK" value={profile.cik} />
              <MetricBadge label="SIC Code" value={profile.sic_code} />
              <MetricBadge label="Fiscal Year End" value={formatFiscalYearEnd(profile.fiscal_year_end)} />
              <MetricBadge label="Industry" value={profile.industry} />
            </div>
            {profile.business_address && (
              <div className="mt-2 text-xs text-gray-500 font-mono">
                {[profile.business_address.street, profile.business_address.city, profile.business_address.state, profile.business_address.zip].filter(Boolean).join(', ')}
              </div>
            )}
          </div>
        )}

        {/* ── Compare: second company search ───────────────────────────────── */}
        {compareMode && profile && jobStatus?.status === 'complete' && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-3">
              Compare with Another Company
            </div>
            <div className="relative">
              <input
                type="text"
                placeholder="Enter company name or ticker to compare"
                value={compareQuery}
                onChange={e => { setCompareQuery(e.target.value); handleCompareSearch(e.target.value); }}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 transition-colors"
              />
              {compareQuery && compareResults.length > 0 && (
                <ul className="absolute z-10 w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg max-h-52 overflow-y-auto shadow-2xl">
                  {compareResults.map(c => (
                    <li
                      key={c.cik}
                      onClick={() => handleCompareSelect(c)}
                      className="px-4 py-3 cursor-pointer hover:bg-gray-800 border-b border-gray-800 last:border-b-0 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-100">{c.title}</span>
                        <span className="text-xs bg-green-900/40 text-green-300 border border-green-800 px-2 py-0.5 rounded font-mono">
                          {c.ticker}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {compareCompany && (
              <div className="mt-3">
                <div className="text-xs text-gray-500 mb-2">
                  Comparing: <span className="text-blue-400 font-mono">{profile.ticker}</span>
                  {' vs '}
                  <span className="text-green-400 font-mono">{compareCompany.ticker}</span>
                </div>
                {compareJobStatus?.status === 'processing' && (
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="flex gap-1">
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce [animation-delay:300ms]" />
                      </div>
                      <span className="text-xs text-green-400">Fetching & Summarizing {compareCompany.ticker} {formType} filing...</span>
                    </div>
                    <ProgressBar progress={compareJobStatus.progress} />
                  </div>
                )}
                {comparisonLoading && (
                  <div className="text-xs text-purple-400 animate-pulse">Generating key differences...</div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Compare company info card ─────────────────────────────────────── */}
        {compareCompany && compareJobStatus?.status === 'complete' && compareJobStatus?.result?.company_profile && (
          <div className="bg-gray-900 border border-green-800/40 border-l-2 border-l-green-500 rounded-xl p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-[10px] text-green-500 uppercase tracking-wider mb-1">Comparing Against</div>
                <h2 className="text-xl font-bold text-gray-50 mb-1">{compareJobStatus.result.company_profile.name}</h2>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs bg-green-900/40 text-green-300 border border-green-800 px-2 py-0.5 rounded font-mono">
                    {compareJobStatus.result.company_profile.ticker}
                  </span>
                  {compareJobStatus.result.company_profile.exchanges?.map(ex => (
                    <span key={ex} className="text-xs text-gray-500">{ex}</span>
                  ))}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <MetricBadge label="CIK" value={compareJobStatus.result.company_profile.cik} />
              <MetricBadge label="SIC Code" value={compareJobStatus.result.company_profile.sic_code} />
              <MetricBadge label="Fiscal Year End" value={formatFiscalYearEnd(compareJobStatus.result.company_profile.fiscal_year_end)} />
              <MetricBadge label="Industry" value={compareJobStatus.result.company_profile.industry} />
            </div>
            {compareJobStatus.result.company_profile.business_address && (
              <div className="mt-2 text-xs text-gray-500 font-mono">
                {[compareJobStatus.result.company_profile.business_address.street, compareJobStatus.result.company_profile.business_address.city, compareJobStatus.result.company_profile.business_address.state, compareJobStatus.result.company_profile.business_address.zip].filter(Boolean).join(', ')}
              </div>
            )}
          </div>
        )}

        {/* ── Filing Summary (accordion) ────────────────────────────────────── */}
        {summary && jobStatus?.status === 'complete' && !compareCompany && (
          <SummaryAccordionGroup
            summary={summary}
            formType={result?.form_type || formType}
            onExportPdf={handleExportPdf}
          />
        )}

        {/* ── News Sentiment ───────────────────────────────────────────────── */}
        {(newsLoading || newsSentiment) && jobStatus?.status === 'complete' && !compareCompany && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-1 h-5 bg-yellow-500 rounded" />
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Live News Sentiment</h3>
              {newsSentiment && newsSentiment.sentiment !== 'unavailable' && (
                <span className={`text-xs px-2 py-0.5 rounded border font-semibold ${
                  newsSentiment.sentiment === 'positive'
                    ? 'bg-green-900/40 text-green-300 border-green-700'
                    : newsSentiment.sentiment === 'negative'
                    ? 'bg-red-900/40 text-red-300 border-red-700'
                    : 'bg-gray-800 text-gray-400 border-gray-700'
                }`}>
                  {newsSentiment.sentiment === 'positive' ? '📈 Positive' : newsSentiment.sentiment === 'negative' ? '📉 Negative' : '↔️ Neutral'}
                </span>
              )}
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              {newsLoading && !newsSentiment && (
                <div className="text-sm text-gray-500 animate-pulse">Fetching latest headlines...</div>
              )}
              {newsSentiment && (
                <div className="space-y-3">
                  {newsSentiment.summary && (
                    <div className="prose prose-invert prose-base max-w-none text-gray-200 leading-relaxed">
                      <ReactMarkdown>{newsSentiment.summary}</ReactMarkdown>
                    </div>
                  )}
                  {newsSentiment.headlines?.length > 0 && (
                    <div className="border-t border-gray-800 pt-3">
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Recent Headlines</div>
                      <ul className="space-y-1.5">
                        {newsSentiment.headlines.map((h, i) => (
                          <li key={i} className="text-xs text-gray-400 flex items-start gap-2">
                            <span className="text-gray-600 mt-0.5 shrink-0">•</span>
                            {h.link ? (
                              <a href={h.link} target="_blank" rel="noopener noreferrer"
                                className="hover:text-blue-400 transition-colors">{h.title}</a>
                            ) : (
                              <span>{h.title}</span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Comparison Results ────────────────────────────────────────────── */}
        {comparisonData && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-1 h-5 bg-green-500 rounded" />
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">
                Comparison: <span className="text-blue-400">{comparisonData.ticker1}</span>
                {' vs '}
                <span className="text-green-400">{comparisonData.ticker2}</span>
              </h3>
            </div>

            {/* Key Differences LLM summary */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-3">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Key Differences & Similarities</div>
              <div className="prose prose-invert prose-base max-w-none text-gray-200 leading-relaxed">
                <ReactMarkdown>{comparisonData.key_differences}</ReactMarkdown>
              </div>
            </div>

            {/* Side-by-side section accordions with Expand/Collapse/Export controls */}
            <CompareAccordionGroup
              comparisonData={comparisonData}
              onExportPdf={handleExportComparePdf}
            />
          </div>
        )}

        {/* ── Financial Charts ─────────────────────────────────────────────── */}
        {charts && jobStatus?.status === 'complete' && !compareCompany && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-1 h-5 bg-green-500 rounded" />
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">
                Financial Statements: 5-Year Analysis
              </h3>
            </div>

            {/* Chart tabs */}
            <div className="flex border-b border-gray-800 mb-3">
              {[
                { key: 'income_statement', label: 'Income Statement' },
                { key: 'balance_sheet', label: 'Balance Sheet' },
                { key: 'cash_flow', label: 'Cash Flow' },
              ].map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setActiveChart(tab.key)}
                  className={`px-4 py-2 text-xs font-semibold transition-colors relative ${
                    activeChart === tab.key
                      ? 'text-blue-400'
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {tab.label}
                  {activeChart === tab.key && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
                  )}
                </button>
              ))}
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              {charts[activeChart] ? (
                <img
                  src={`data:image/png;base64,${charts[activeChart]}`}
                  alt={`${activeChart} chart`}
                  className="w-full h-auto rounded"
                />
              ) : (
                <div className="text-center text-gray-600 py-12 text-sm">Chart Not Available</div>
              )}
            </div>
          </div>
        )}

        {/* ── Q&A Section ──────────────────────────────────────────────────── */}
        {summary && jobStatus?.status === 'complete' && (
          <div>
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              <div className="w-1 h-5 bg-purple-500 rounded" />
              <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Ask Questions</h3>
              {compareCompany && compareJobStatus?.status === 'complete' && (
                <div className="flex gap-1 border border-gray-700 rounded overflow-hidden ml-auto">
                  <button
                    onClick={() => { setQaCompany('primary'); setChatHistory([]); setStreamingContent(''); }}
                    className={`px-3 py-1 text-xs font-mono transition-colors ${qaCompany === 'primary' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
                  >
                    {selectedCompany?.ticker}
                  </button>
                  <button
                    onClick={() => { setQaCompany('compare'); setChatHistory([]); setStreamingContent(''); }}
                    className={`px-3 py-1 text-xs font-mono transition-colors ${qaCompany === 'compare' ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
                  >
                    {compareCompany.ticker}
                  </button>
                </div>
              )}
            </div>

            {/* Chat history */}
            {chatHistory.length > 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-3 max-h-96 overflow-y-auto">
                {chatHistory.map((msg, i) => (
                  <ChatBubble
                    key={i}
                    role={msg.role}
                    content={msg.content}
                    cited={msg.cited}
                    source={msg.source}
                    filingType={result?.form_type || formType}
                  />
                ))}
                {/* Streaming cursor */}
                {isStreaming && streamingContent && (
                  <div className="flex justify-start mb-3">
                    <div className="max-w-[85%] px-4 py-3 rounded-xl rounded-bl-none bg-gray-800 text-gray-200 border border-gray-700 text-sm">
                      <div className="prose prose-invert prose-base max-w-none leading-relaxed text-gray-100">
                        <ReactMarkdown>{streamingContent}</ReactMarkdown>
                      </div>
                      <span className="inline-block w-1.5 h-3.5 bg-blue-400 ml-0.5 animate-pulse rounded-sm" />
                    </div>
                  </div>
                )}
                {isStreaming && !streamingContent && (
                  <div className="flex justify-start mb-3">
                    <div className="px-4 py-3 rounded-xl rounded-bl-none bg-gray-800 border border-gray-700">
                      <div className="flex gap-1">
                        <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:300ms]" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}

            {/* Suggestion chips */}
            <div className="flex flex-wrap gap-2 mb-3">
              {SUGGESTION_CHIPS.map((chip, i) => (
                <button
                  key={i}
                  onClick={() => handleAsk(chip)}
                  disabled={isStreaming}
                  className="text-xs bg-gray-900 text-gray-400 border border-gray-700/80 px-3 py-1.5 rounded-full hover:bg-blue-950/60 hover:text-blue-300 hover:border-blue-700/60 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {chip}
                </button>
              ))}
            </div>

            {/* Input */}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="e.g., What were the main revenue drivers last year?"
                value={qaInput}
                onChange={e => setQaInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(qaInput); } }}
                disabled={isStreaming}
                className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 disabled:opacity-50 transition-colors"
              />
              <button
                onClick={() => handleAsk(qaInput)}
                disabled={isStreaming || !qaInput.trim()}
                className="bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg text-sm font-semibold transition-colors"
              >
                {isStreaming ? '…' : 'Ask'}
              </button>
            </div>

            <p className="text-[11px] text-gray-600 mt-2">
              You can ask follow-ups like "Explain that further" or "What about 2022?"
            </p>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800/60 mt-12 py-4 px-6 text-center">
        <p className="text-[11px] text-gray-700">
          FinSight AI · Data sourced from SEC EDGAR · For informational purposes only, not financial advice
        </p>
      </footer>
    </div>
    <Analytics />
    </>
  );
}
