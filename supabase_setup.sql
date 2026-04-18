-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)

-- Users table (login)
create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    username text unique not null,
    password_hash text not null,
    created_at timestamptz default now()
);

-- Simulation results (live sentiment per tweet batch)
create table if not exists sim_results (
    id bigserial primary key,
    ticker text not null,
    model text not null,
    trading_date date not null,
    label text not null,
    sentiment_score float,
    buy_pct float,
    sell_pct float,
    hold_pct float,
    no_opinion_pct float,
    tweet_volume int,
    created_at timestamptz default now()
);

-- Simulation state (one row per model, tracks replay position)
create table if not exists sim_state (
    model text primary key,
    position int default 0,
    status text default 'paused',   -- running / paused / finished
    speed_seconds int default 30,
    updated_at timestamptz default now()
);

-- Seed default simulation states
insert into sim_state (model, position, status, speed_seconds)
values
    ('vader',   0, 'paused', 30),
    ('finbert', 0, 'paused', 30),
    ('gpt',     0, 'paused', 30)
on conflict (model) do nothing;

-- Index for fast dashboard queries
create index if not exists idx_sim_results_ticker_model on sim_results(ticker, model);
create index if not exists idx_sim_results_created on sim_results(created_at desc);
