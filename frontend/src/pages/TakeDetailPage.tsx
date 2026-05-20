import { useCallback, useEffect, useState } from "react";
import { api, type TakeDetail } from "../api/client";
import { connectEvents } from "../api/events";
import TakeDetailView from "./TakeDetailView";

type Props = {
  takeId: string;
};

export default function TakeDetailPage({ takeId }: Props) {
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.take(takeId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load take");
    }
  }, [takeId]);

  useEffect(() => {
    void load();
    const source = connectEvents(() => void load(), () => undefined);
    const timer = window.setInterval(() => void load(), 2000);
    return () => {
      source?.close();
      window.clearInterval(timer);
    };
  }, [load]);

  if (error) {
    return <main className="page-pad"><div className="empty-state">{error}</div></main>;
  }
  return <main className="page-pad"><TakeDetailView detail={detail} /></main>;
}
