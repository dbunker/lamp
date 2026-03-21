%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.191s
%% Conflict analysis                       : 0.569s
%%   - Positive Examples                   : 0.569s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.13s
%% Total                                   : 0.931s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d0(V1).
d1(V1) :- d2(V1); not d4(V1).
d0(V1) :- d3(V1); not d2(V1).
d3(V1) :- d1(V1); not d5(V1).

