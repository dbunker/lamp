%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.156s
%% Conflict analysis                       : 0.362s
%%   - Positive Examples                   : 0.362s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.116s
%% Total                                   : 0.666s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d0(V1) :- d2(V1).
d1(V1) :- d2(V1).
d3(V1) :- d0(V1).

