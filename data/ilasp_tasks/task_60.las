#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o4).

#modeh(d1(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).

#pos(eg1, {
  d1(o2)
}, {
  d1(o0), d1(o4)
}, {
  d0(o0).
  d2(o2).
  d3(o0).
  d3(o2).
  d3(o4).
}).