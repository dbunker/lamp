%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.099s
%% Conflict analysis                       : 24.687s
%%   - Positive Examples                   : 24.687s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 10.242s
%% Total                                   : 38.453s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d4(V1).
d9(V1) :- d4(V1).
d2(V1) :- d4(V1).
d4(V1) :- d0(V1).
d7(V1) :- d4(V1).
d3(V1) :- d4(V1).

