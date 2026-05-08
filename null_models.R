library(tidyverse)
library(igraph)
library(tidygraph)

shuffle_bipartite_incidence <-
  function(g,
           keep_vertex_names = TRUE) {

    inc_mat <-
      igraph::as_incidence_matrix(
        g,
        sparse = FALSE
      )


    shuffled_inc_mat <-
      matrix(
        sample(as.vector(inc_mat)),
        nrow = nrow(inc_mat),
        ncol = ncol(inc_mat)
      )

    if (keep_vertex_names) {
      rownames(shuffled_inc_mat) <- rownames(inc_mat)
      colnames(shuffled_inc_mat) <- colnames(inc_mat)
    }

    shuffled_g <-
      igraph::graph_from_incidence_matrix(
        shuffled_inc_mat,
        directed = igraph::is_directed(g)
      )

    shuffled_g
  }


# read data ----

g_no <- read_graph(file = "data/breadbox_bipartite_nonIBD_q9995_KEEPALLNODES_FULL.graphml", format = "graphml")
V(g_no)$type <- V(g_no)$bipartite

g_uc <- read_graph(file = "data/breadbox_bipartite_UC_q9995_KEEPALLNODES_FULL.graphml", format = "graphml")
V(g_uc)$type <- V(g_uc)$bipartite

g_cd <- read_graph(file = "data/breadbox_bipartite_CD_q9995_KEEPALLNODES_FULL.graphml", format = "graphml")
V(g_cd)$type <- V(g_cd)$bipartite

# generate null models (basic rewiring) ----
random_no <-
  lapply(1:100, function(i){
    set.seed(i)
    shuffle_bipartite_incidence(
      g = g_no,
      keep_vertex_names = T
      )
  })

random_cd <-
  lapply(1:100, function(i){
    set.seed(i)
    shuffle_bipartite_incidence(
      g = g_cd,
      keep_vertex_names = T
    )
  })

random_uc <-
  lapply(1:100, function(i){
    set.seed(i)
    shuffle_bipartite_incidence(
      g = g_uc,
      keep_vertex_names = T
    )
  })

# ease of use writeout ----

list(random_no = random_no,
     random_cd = random_cd,
     random_uc = random_uc

     ) |>
  write_rds("null_model.rds")

# node naming ----

transfer_ids_to_graph_list <-
  function(g_original, graph_list) {

    id_map <-
      tibble::tibble(
        name = as.character(seq_along(igraph::V(g_original))),
        id = igraph::V(g_original)$id
      )

    purrr::map(
      graph_list,
      function(g) {

        vertex_df <-
          tibble::tibble(
            name = as.character(igraph::V(g)$name)
          ) |>
          dplyr::left_join(
            id_map,
            by = "name"
          )

        igraph::V(g)$id <-
          vertex_df$id

        g
      }
    )
  }

random_no.id <-
  transfer_ids_to_graph_list(g_original = g_no,
                           graph_list = random_no
                           )

random_uc.id <-
  transfer_ids_to_graph_list(g_original = g_uc,
                             graph_list = random_uc
  )

random_cd.id <-
  transfer_ids_to_graph_list(g_original = g_cd,
                             graph_list = random_cd
  )
# degree calculation and tidygraphing ----

random_no.id <-
  lapply(random_no.id, FUN = function(i){
  i |>
    as_tbl_graph() |>
    mutate(k = centrality_degree())
})


random_uc.id <-
  lapply(random_uc.id, FUN = function(i){
    i |>
      as_tbl_graph() |>
      mutate(k = centrality_degree())
  })

random_cd.id <-
  lapply(random_cd.id, FUN = function(i){
    i |>
      as_tbl_graph() |>
      mutate(k = centrality_degree())
  })

## calculate avg degree and sd per node ----

random_no.k <-
  random_no.id |>
  lapply(FUN = as_tibble) |>
  bind_rows() |>
  group_by(id, type) |>
  summarise(mean_k = mean(k, na.rm=T),
            sd_k   = sd(k, na.rm = T)
            )

random_cd.k <-
  random_cd.id |>
  lapply(FUN = as_tibble) |>
  bind_rows() |>
  group_by(id, type) |>
  summarise(mean_k = mean(k, na.rm=T),
            sd_k   = sd(k, na.rm = T)
  )

random_uc.k <-
  random_uc.id |>
  lapply(FUN = as_tibble) |>
  bind_rows() |>
  group_by(id, type) |>
  summarise(mean_k = mean(k, na.rm=T),
            sd_k   = sd(k, na.rm = T)
  )

# join with empirical data ----

tbl.g_no <-
  g_no |>
  as_tbl_graph() |>
  select(id, type) |>
  mutate(type = as.logical(type)) |>
  mutate(k = centrality_degree()) |>
  as_tibble() |>
  left_join(y = random_no.k) |>
  mutate(z_score = (k - mean_k)/sd_k)


tbl.g_uc <-
  g_uc |>
  as_tbl_graph() |>
  select(id, type) |>
  mutate(type = as.logical(type)) |>
  mutate(k = centrality_degree()) |>
  as_tibble() |>
  left_join(y = random_uc.k) |>
  mutate(z_score = (k - mean_k)/sd_k)

tbl.g_cd <-
  g_cd |>
  as_tbl_graph() |>
  select(id, type) |>
  mutate(type = as.logical(type)) |>
  mutate(k = centrality_degree()) |>
  as_tibble() |>
  left_join(y = random_cd.k) |>
  mutate(z_score = (k - mean_k)/sd_k)

#writeouts ----

tbl.g_no |>
  write_csv(file = "no_null_model_degree.csv")

tbl.g_cd |>
  write_csv(file = "cd_null_model_degree.csv")

tbl.g_uc |>
  write_csv(file = "uc_null_model_degree.csv")

#

null_no.global <-
  lapply(random_no.id, function(i){

  ii <-
    i |>
    as_tbl_graph() |>
    mutate(k = centrality_degree())

  micro_connected <-
   ii |>
    filter(!type) |>
    filter(k > 0) |>
    as_tibble() |>
    nrow()


  gene_connected <-
    ii |>
    filter(type) |>
    filter(k > 0) |>
    as_tibble() |>
    nrow()

  components <-
    ii |> components()

  no_components <-
    components$no

  size_lcc <-
    max(components$csize, na.rm = T)

  tibble(nodes_micro_connected = micro_connected,
         nodes_gene_connected  = gene_connected,
         no_components   = no_components,
         size_lcc        = no_components

         )
}) |>
  bind_rows()


null_uc.global <-
  lapply(random_uc.id, function(i){

    ii <-
      i |>
      as_tbl_graph() |>
      mutate(k = centrality_degree())

    micro_connected <-
      ii |>
      filter(!type) |>
      filter(k > 0) |>
      as_tibble() |>
      nrow()


    gene_connected <-
      ii |>
      filter(type) |>
      filter(k > 0) |>
      as_tibble() |>
      nrow()

    components <-
      ii |> components()

    no_components <-
      components$no

    size_lcc <-
      max(components$csize, na.rm = T)

    tibble(nodes_micro_connected = micro_connected,
           nodes_gene_connected  = gene_connected,
           no_components   = no_components,
           size_lcc        = no_components

    )
  }) |>
  bind_rows()

null_cd.global <-
  lapply(random_cd.id, function(i){

    ii <-
      i |>
      as_tbl_graph() |>
      mutate(k = centrality_degree())

    micro_connected <-
      ii |>
      filter(!type) |>
      filter(k > 0) |>
      as_tibble() |>
      nrow()


    gene_connected <-
      ii |>
      filter(type) |>
      filter(k > 0) |>
      as_tibble() |>
      nrow()

    components <-
      ii |> components()

    no_components <-
      components$no

    size_lcc <-
      max(components$csize, na.rm = T)

    tibble(nodes_micro_connected = micro_connected,
           nodes_gene_connected  = gene_connected,
           no_components   = no_components,
           size_lcc        = no_components

    )
  }) |>
  bind_rows()

# convenience function to summarize data ----

summarize_function <-
  function(x){
    x |>
    dplyr::summarise(
      dplyr::across(
        where(is.numeric),
        list(
          mean = ~mean(.x, na.rm = TRUE),
          sd   = ~sd(.x, na.rm = TRUE)
        ),
        .names = "{.col}_{.fn}"
      )
    )

  }

tibble_summary_globals <-
  list(non_ibd = null_no.global,
     uc = null_uc.global,
     cd = null_cd.global) |>
  lapply(FUN = summarize_function) |>
  bind_rows(.id = "phenotype")


