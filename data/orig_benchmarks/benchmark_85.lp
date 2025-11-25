d2(o3).
d4(o0).
d1(o0).
d3(o1).

d3(X) :- d2(X), d4(X).
d4(X) :- d1(X), d3(X).
d3(X) :- d1(X), d4(X).