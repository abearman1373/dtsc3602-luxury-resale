-- Run once in the Supabase SQL editor before scripts/load_supabase.py.
-- The reserved test rows are NOT loaded, so the dashboard can never show them.

drop table if exists listings_train;
create table listings_train (
  listing_id             bigint primary key,
  brand                  text,
  brand_group            text,
  model                  text,
  item_type              text,
  item_group             text,
  product_type           text,
  condition              text,
  condition_rank         int,
  item_age_years         double precision,
  age_band               text,
  age_available          boolean,
  age_censored           boolean,
  material_group         text,
  n_extras               int,
  has_dust_bag           boolean,
  has_strap              boolean,
  estimated_retail_price double precision,
  asking_price           double precision,
  asking_price_ratio     double precision,
  listed_at              date,
  days_listed            int,
  is_recent              boolean,
  available              boolean,
  cv_fold                int
);

-- All 15,888 SouthPark listings, without prices, for the coverage / representativeness page.
drop table if exists southpark_inventory;
create table southpark_inventory (
  listing_id       bigint primary key,
  brand            text,
  item_type        text,
  condition        text,
  has_target       boolean,
  dropship_partner boolean,
  is_recent        boolean,
  listed_at        date
);

-- Dashboard reads with the anon key: allow read-only access.
alter table listings_train enable row level security;
alter table southpark_inventory enable row level security;
create policy "public read" on listings_train for select using (true);
create policy "public read" on southpark_inventory for select using (true);
