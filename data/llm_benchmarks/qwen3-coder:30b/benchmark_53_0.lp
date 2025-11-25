d0(o4).
d3(o4).
d3(o1).
d1(o4).

d1(X) :- d0(X), not d2(X).
d3(X) :- d0(X), not d1(X).
d2(X) :- d3(X), not d1(X), not d0(X).