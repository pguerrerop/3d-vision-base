import { isPartialRerunExecutionMode, partialRunNoticeText } from "./partialRunNoticeModel";

type Props = {
  executionMode: string | null | undefined;
  parentRunId: string | null | undefined;
  boundaryStageId: string | null | undefined;
};

export default function PartialRunNoticeBanner({ executionMode, parentRunId, boundaryStageId }: Props) {
  if (!isPartialRerunExecutionMode(executionMode)) return null;
  return (
    <div className="partial-run-notice-banner empty-state compact">
      {partialRunNoticeText({ parentRunId, boundaryStageId })}
    </div>
  );
}
