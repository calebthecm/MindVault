import { NextResponse } from "next/server";

// Proxy PyPI's version data — always reflects the latest published release
// without needing any manual updates to this API.
export async function GET() {
  try {
    const res = await fetch("https://pypi.org/pypi/mindvault/json", {
      next: { revalidate: 300 }, // cache for 5 minutes
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: "upstream_unavailable" },
        { status: 502 }
      );
    }

    const data = await res.json();
    const version: string = data?.info?.version ?? "";

    if (!version) {
      return NextResponse.json({ error: "version_not_found" }, { status: 502 });
    }

    return NextResponse.json({ version });
  } catch {
    return NextResponse.json({ error: "fetch_failed" }, { status: 502 });
  }
}
