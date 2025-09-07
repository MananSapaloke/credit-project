'use client';

import { SessionProvider } from 'next-auth/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { SWRConfig } from 'swr';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // 1 minute
        cacheTime: 5 * 60 * 1000, // 5 minutes
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  }));

  return (
    <SessionProvider>
      <QueryClientProvider client={queryClient}>
        <SWRConfig
          value={{
            fetcher: (url: string) => fetch(url).then((res) => res.json()),
            revalidateOnFocus: false,
            revalidateOnReconnect: true,
            dedupingInterval: 2000,
          }}
        >
          {children}
        </SWRConfig>
      </QueryClientProvider>
    </SessionProvider>
  );
}
