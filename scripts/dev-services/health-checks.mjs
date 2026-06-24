export async function checkServiceHealth(service) {
  if (!service.port || !Array.isArray(service.healthPaths) || service.healthPaths.length === 0) {
    return { ok: null, checks: [] };
  }

  const checks = [];
  const hosts = Array.from(new Set((service.hosts ?? ["127.0.0.1", "localhost"]).filter(Boolean)));
  for (const healthPath of service.healthPaths) {
    for (const host of hosts) {
      const url = `http://${host}:${service.port}${healthPath}`;
      try {
        const response = await fetch(url, {
          method: "GET",
          redirect: "manual",
          signal: AbortSignal.timeout(1500),
        });
        checks.push({
          url,
          ok: response.status >= 200 && response.status < 400,
          status: response.status,
        });
        break;
      } catch (error) {
        checks.push({
          url,
          ok: null,
          status: null,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  const reachable = checks.filter((item) => item.ok !== null);
  return {
    ok: reachable.length === 0 ? null : reachable.every((item) => item.ok),
    checks,
  };
}
