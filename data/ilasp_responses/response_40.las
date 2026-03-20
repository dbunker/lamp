%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.164s
%% Conflict analysis                       : 0.39s
%%   - Positive Examples                   : 0.39s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.102s
%% Total                                   : 0.687s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d3(V1).
d0(V1) :- d2(V1).
d1(V1) :- d4(V1).
d0(V1) :- d5(V1); not d4(V1).

