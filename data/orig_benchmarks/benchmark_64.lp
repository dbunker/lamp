d2(o4).
d3(o0).
d2(o1).
d3(o2).
d0(o4).

d3(X) :- not d1(X), d2(X), d0(X).
d1(X) :- not d0(X), d2(X), d3(X).