%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.376s
%% Conflict analysis                       : 23.131s
%%   - Positive Examples                   : 23.131s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 9.358s
%% Total                                   : 36.583s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d2(V1).
d4(V1) :- d3(V1).
d7(V1) :- d4(V1).
d1(V1) :- d4(V1).
d8(V1) :- d5(V1).
d0(V1) :- d5(V1).

