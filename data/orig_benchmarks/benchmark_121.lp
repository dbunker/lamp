d4(o1).
d4(o3).
d3(o3).
d1(o1).

d3(X) :- d1(X), d4(X).
d1(X) :- d2(X), d3(X).
d2(X) :- d1(X), d4(X).